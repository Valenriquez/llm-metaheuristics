import ollama
import chromadb
import os
import datetime
import subprocess
import logging
import re
import pathlib
from pathlib import Path
import json
import numpy as np
import math
from chromadb.utils import embedding_functions
from sklearn.decomposition import PCA
import scipy.stats as stats
import numpy as np
from scipy.stats import mannwhitneyu

"""
Uses: ollama model: "qwen2.5-coder:latest"
Uses: embedding model: "all-minilm:latest"

- Creates the metaheuristic (exploration function)
- Runs metaheuristic and calculates performance with fitness
- Tunes metaheuristic with Optuna
- Re-Runs metaheuristic and calculates performance with fitness
- Saves its' parameters and perfomance with ChromaDB (refinement function)


- Creates the feedback collection:
    parameters and performance = 
    self.feedback_collection.add(
        ids=[f"iteration_{number_iteration}"],
        documents=[parameters and performance],
        metadatas={"source": 'operator_name_(1)_operator_name_(2)_operator_name_(3)_1'}
    )

- Get ALL values of the feedback collection
- selected_number_of_iterations will be calculated: number_iterations / selected_number_of_iterations
- After {selected_number_of_iterations} ask Ollama best (smallest)  performance encountered

When refining more a metaheuristic (with new metaheuristic generation each time)
    - {break_tries} : variable to limit the tries of corrected metaheuristic generation
    - Added:  self.performance_found < self.best_performance:
            - Which means that will run till the output is correct and till the fitness is better than the previous one. 
            Although there is a break after the {break_tries} try. 
    - (?) After creation must verify that the next metaheuristic generation will have a similar space (?) 

Feedback: {data_feedback} (if none are provided, you may skip this part).
"""

class GerateMetaheuristic:
    def __init__(self, problem_id, dimensions, max_iterations, model="qwen2.5-coder:latest", model_embed="all-minilm:latest"):
        self.model = model
        self.model_embed = model_embed
        self.max_iterations = max_iterations

        ## METAHEURISTIC CREATION
        self.problem_id = problem_id
        self.dimensions = dimensions
        self.hyperparameters = ""
        self.best_performance = 500
        self.performance_found = 200
        self.made_metaheuristic = ""
        self.the_bool = True

        self.old_losses = []
        self.new_losses = []
        self.history_append = []

        ## OUTPUT INFO
        self.folder_name = ""
        self.extracted_metaheuristic = ""

        ## LOGGER
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        ## ERROR CHECKER
        self.file_result = 1
        self.file_result_error = ""
        self.total_budget = 100 # (?)

        ## NEW APPROACH
        self.break_tries = 8 # for Optuna Enhancement only
        self.num_of_agents = 10 * self.dimensions + math.floor(20 * math.log(self.dimensions + 1))
        self.standard_dev = 0

        ## FOR OPTUNA
        self.file_contents = "" 

        ## COLLECTIONS
        #chroma_client = chromadb.PersistentClient("vectordb")
        chroma_client = chromadb.Client()
        google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyClnhvj-6aQdDS2qcheEoep2SiCUXvQz-I")
        #sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-en-v1.5"
        )

        #chroma_client.delete_collection(name="operators_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
        #chroma_client.delete_collection(name="metaheuristic_template_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
        #chroma_client.delete_collection(name="feedback_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
        #chroma_client.delete_collection(name="optuna_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible


        self.operators_collection = chroma_client.create_collection(name="operators_collection", embedding_function=google_ef)
        self.metaheuristic_template_collection = chroma_client.create_collection(name="metaheuristic_template_collection", embedding_function=google_ef)
        self.feedback_collection = chroma_client.create_collection(name="feedback_collection",  embedding_function=sentence_transformer_ef)
        self.optuna_collection = chroma_client.create_collection(name="ioh_optuna_builder", embedding_function=google_ef)

        self.creating_collection("operators_collection", self.operators_collection)
        self.creating_collection("metaheuristic_template_collection", self.metaheuristic_template_collection)
        self.creating_collection("ioh_optuna_builder", self.optuna_collection)

        self.prompt = f"""You are a highly skilled computer scientist in the field of natural computing. Your task is to design a metaheuristic algorithm, 
        you should only use the information that was provided to you. 
        Remember that when writing the operator's names, they should be ALL in LOWER CASE AND WITH A '_' instead of typing a space. 
        """""
    
    """
    creating_collection: Will create the ChromaDB collection
    """
    def creating_collection(self, name_of_folder, name_of_collection):
        directory = os.path.join(os.path.dirname(__file__), name_of_folder)
        for d in os.listdir(directory):
            file_path = os.path.join(directory, d)
            if os.path.isfile(file_path):  # Check if it's a file
                file_content = self.read_file(file_path)
                response = ollama.embeddings(model=self.model_embed, prompt=file_content)
                embedding = response.get("embedding")
                if embedding:
                    name_of_collection.add(
                        ids=[d],
                        embeddings=[embedding],
                        documents=[file_content],
                        metadatas=[{"hnsw:space": "cosine"} ]
                    )
                    print(f"Added {d} to the collection {name_of_collection}")
                else:
                    print(f"Warning: Empty embedding generated for {d}")
            else:  # If it's not a file, skip it
                continue

    """
    creating_operators_collection: Will create the ChromaDB collection
    """
    def creating_operators_collection(self, name_of_folder, name_of_collection):
        directory = os.path.join(os.path.dirname(__file__), name_of_folder)
        for d in os.listdir(directory):
            file_path = os.path.join(directory, d)
            if os.path.isfile(file_path):  # Check if it's a file
                file_content = self.read_file(file_path)
                # Split the content into blocks by double newlines
                blocks = file_content.split("\n\n")
                # Clean up and store blocks
                documents = [block.strip() for block in blocks if block.strip()]
                # Print results for verification
                for i, block in enumerate(documents, start=1):
                    print(f"{'-' * 40}")
                    response = ollama.embeddings(model="all-minilm:latest", prompt=block)
                    embedding = response.get("embedding")

                    # Add embedding to the collection if valid
                    if embedding:
                        name_of_collection.add(
                            ids=[f"operator: {i}"],
                            embeddings=[embedding],
                            documents=[block],
                            metadatas=[{"hnsw:space": "cosine"}]
                        )
                        print(f"Added {i} to the collection")
                    else:
                        print(f"Warning: Empty embedding generated for {i}")


    """
    read_file:  Read file
    """  
    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    """
    transform_embeddings:   
    """ 
    def transform_embeddings(embeddings, target_dim=1024):
        pca = PCA(n_components=target_dim)
        transformed_embeddings = pca.fit_transform(embeddings)
        return transformed_embeddings

        
    """
    extract_code_from_code_with_optuna:  Modify metaheuristic to make optuna format 
    """                

    def extract_code_from_code_with_optuna(self, code_file):
        pattern = r'heur\s*=\s*\[(.*?)\]' 
        match = re.search(pattern, code_file, re.DOTALL)
 
        if match:
            extracted_content = match.group(1).strip()  
            print("MATCH", extracted_content)

            output = ollama.generate(
            model = self.model,
            prompt = f"""Modify this metaheuristic: {extracted_content}, DO NOT ADD any new parameters. 
                         You must make the following changes based on the parameter type:

                        1. **If the parameter provides a numeric value:**
                        - Change it to the format:
                            name_variable = trial.suggest_float('name_variable', lower_limit, upper_limit)
                        - Use appropriate ranges for `lower_limit` and `upper_limit`. 
                            - **Rules for ranges:**
                            - For **radius**, the maximum is 0.9.
                            - For **angle**, the maximum is 25.
                            - For **swarm_conf** or **self_conf**, the maximum is 3.
                        - **Incorrect Example:**
                            'name_variable': trial.suggest_categorical('name_variable', ['2.54'])
                        - **Correct Example:**
                            'beta': trial.suggest_float('beta', 1.5, 4.5),
                        - Always include a comma after the modified parameter.

                        2. **If the parameter provides a category:**
                        - Modify it to the format:
                            'category_name': trial.suggest_categorical('category_name', ['option_1', 'option_2', 'option_3'])
                        - Include at least **three options**. ALWAYS USE MORE THAN ONE option.
                        - **Incorrect Example:**
                            'category_name': trial.suggest_categorical('category_name', ['option_1'])
                        - **Correct Example:**
                            'distribution': trial.suggest_categorical('distribution', ['uniform', 'gaussian','levy'])


                        3. **Rules for specific category types:**
                        - "version": "inertial", "constriction"
                        - "distribution": "uniform", "gaussian", "levy"
                        - "pairing": "rank", "cost", "random", "tournament_2_100"
                        - "crossover": "single", "two", "uniform", "blend", "linear_0.5_0.5"
                        - "expression": "rand", "best", "current", "current-to-best", "rand-to-best", "rand-to-best-and-current"

                        4. **General Guidelines:**
                        - Do not change anything else in the metaheuristic. Only modify the parameters as instructed.
                        - Do not add extra words, explanations, or paragraphs. Follow these instructions strictly.
                        """
            ) 
            # Deleted from the prompt: , take a look to the other parameters provided for the operators and selectors: {optuna_data}
            #print("is it doing it wrong", output['response'])
            return output['response']
        else:
            return None


    """
    get_preferential_values:  Gets the best parameters FROM OPTUNA
    """
    def get_preferential_values(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code_file = file.read()
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None, None

        # Regex patterns
        hyperparameters_pattern = r"Mejores hiperparámetros encontrados:\n({.*?})"
        #performance_pattern = r"Mejor rendimiento encontrado:\n([\d.]+)"

        hyperparameters_match = re.search(hyperparameters_pattern, code_file, re.DOTALL)
        hyperparameters_dict = eval(hyperparameters_match.group(1)) if hyperparameters_match else None

        #performance_match = re.search(performance_pattern, code_file)
        #performance_found = float(performance_match.group(1)) if performance_match else None
        #print("performance with funcion found", performance_found)

        #         return hyperparameters_dict, performance_found
        return hyperparameters_dict

    def calculate_performance(self, fitness_array):
        # Check if the array has fitness values
        if fitness_array.size > 0:  
            # Calculate median
            med = np.median(fitness_array)
            # Calculate interquartile range (IQR)
            iqr = np.percentile(fitness_array, 75) - np.percentile(fitness_array, 25)
            # Calculate the performance metric
            performance_metric = med + iqr
            return performance_metric
        else:
            # Return None if the array is empty
            return None
        


    def to_numeric(self, values):
            """Convert a string of values to numeric, ignoring non-numeric entries."""
            numeric_values = []
            for value in values.split():  # Assuming the values are space-separated
                try:
                    numeric_values.append(float(value))  # Attempt conversion to float
                except (ValueError, TypeError):
                    print(f"Skipped invalid value: {value}")  # Debugging invalid values
                    continue
            return numeric_values   
    
    """
    getting_single_metaheuristic_performance:  Gets performance from metaheuristic fitness
    """
    def getting_single_metaheuristic_performance(self, output_folder, number_iteration):
       # Read the results file
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_result_{number_iteration}.txt"
        execution_path = os.path.join(base_path_m, folder_name_m, file_name_m)

        with open(execution_path, 'r') as file:
            content = file.read()

        # Validate and extract content between brackets
        if '[' in content and ']' in content:
            start = content.index('[')
            end = content.index(']')
            raw_values = content[start+1:end]

            # Convert extracted values to numeric
            numeric_values = self.to_numeric(raw_values)

            if len(self.old_losses) == 0: 
                self.old_losses = numeric_values
            else: 
                self.new_losses = numeric_values
                #self.history_append(self.old_losses)  # Keep track of old losses
                #self.old_losses = self.new_losses

            # Convert the numeric list to a NumPy array
            try:
                final_fitness = np.array(self.old_losses, dtype=float)
                if final_fitness.size == 0:
                    raise ValueError("El arreglo está vacío.")

                # Calculate standard deviation and performance
                std_dev = np.std(final_fitness)
                self.performance_found = self.calculate_performance(final_fitness)
                print("Getting: performance_final_fitness", self.performance_found)
            except ValueError as e:
                print(f"Error al procesar array_str: {e}")
                self.performance_found = None
        else:
            print("Error: No se encontraron corchetes en el contenido del archivo.")
            self.performance_found = None

        # Verify if the performance is valid
        if self.performance_found is not None:
            return self.performance_found, std_dev


    """
    get_metaheuristic_performance:  Gets performance from metaheuristic fitness
    """
    def get_metaheuristic_performance(self, output_folder, number_iteration):
        # Helper function to convert values to numeric
       

        # Read the results file
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_result_{number_iteration}.txt"
        execution_path = os.path.join(base_path_m, folder_name_m, file_name_m)

        with open(execution_path, 'r') as file:
            content = file.read()

        # Validate and extract content between brackets
        if '[' in content and ']' in content:
            start = content.index('[')
            end = content.index(']')
            raw_values = content[start+1:end]

            # Convert extracted values to numeric
            numeric_values = self.to_numeric(raw_values)

            if len(self.old_losses) == 0: 
                self.old_losses = numeric_values
            else: 
                self.new_losses = numeric_values
                self.history.append(self.old_losses)  # Keep track of old losses
                self.old_losses = self.new_losses

            # Convert the numeric list to a NumPy array
            try:
                final_fitness = np.array(self.old_losses, dtype=float)
                if final_fitness.size == 0:
                    raise ValueError("El arreglo está vacío.")

                # Calculate standard deviation and performance
                std_dev = np.std(final_fitness)
                self.performance_found = self.calculate_performance(final_fitness)
                print("Getting: performance_final_fitness", self.performance_found)
            except ValueError as e:
                print(f"Error al procesar array_str: {e}")
                self.performance_found = None
        else:
            print("Error: No se encontraron corchetes en el contenido del archivo.")
            self.performance_found = None

        # Verify if the performance is valid
        if self.performance_found is not None:
            return self.performance_found, std_dev

    """
    Exploration:  Will create a metaheuristic and search for new creations. After the first iteration it will look on the feedback for inspiration on better 
    metaheuirsics creation. 

    - Will have access to feedback
    - Will get hyperparameters for the refinement 
    """
    def exploration(self, output_folder, number_iteration):
        self.file_result = 1 # Changing again
        print("i am exploration")
        print("Beginning with exploration number --->",  number_iteration)

        data_feedback = ''
        output_response = None

        # Query for the Operators - - - - - - - - - - - - - - - - - - -
         
        output_operators = ollama.embeddings(
        prompt=f"Remember to write the operators' names",
        model=self.model_embed
        )
        results = self.operators_collection.query(
        query_embeddings=[output_operators["embedding"]],
        n_results=1
        )
        data  = results['documents'][0][0]
        print("operators data:  ", data)
        
        if number_iteration > 1:
            if self.performance_found < self.best_performance:
                self.best_performance = self.performance_found

        # Query for the Operators - - - - - - - - - - - - - - - - - - -
     
          # there won´t be any feedback else:
        if number_iteration > 3:
            # Query for the Feedback - - - - - - - - (asks for the metaheuristics and feedback) - - - - - - - - - - -
            output_feedback = ollama.embeddings(
            prompt="Give me the results with the smallest performance",
            model="bge-m3"
            )
            results = self.feedback_collection.query(
            query_embeddings=[output_feedback["embedding"]],
            n_results=3
            )
            answer = results['documents'][0][0]
            print("data_feedback: Give me the results with the smallest performance", answer)
            # Query for the Feedback - - - - - - - - - - - - - - - - - - -
            output = ollama.generate(
                    model=self.model,
                    prompt = f"""
                   ### INSTRUCTIONS FOR METAHEURISTIC REFINEMENT:
                    1. Analyze the provided metaheuristic data and performance constraints to create a refined metaheuristic.
                    2. Focus on strategies that either **improve performance**, **reduce computational cost**, or achieve a balance between the two.
                    3. Suggest specific modifications to:
                        - Operators
                        - Parameters
                        - Selectors
                    Base your suggestions strictly on the given data without inventing new operators. Variations or combinations of existing operators are acceptable.
                    4. Output the response in **code format only**, avoiding any explanations, markdown, or triple backticks (` ``` `).
                    5. Ensure the metaheuristic design reflects logical, diverse, and effective strategies without exceeding computational constraints.

                    ### CONTEXT:
                    - **Current Iteration**: {number_iteration}
                    - **Best Metaheuristic So Far**: {answer}

                    ### OBJECTIVE:
                    Generate a code-based response that integrates the feedback and refines the metaheuristic to improve the performance.
                    """
                )
            answer = output['response']

            

       # Main Prompt - - - - - - - - - - - - - - - - - - -
        # and (performance_final_fitness < self.best_performance)
        # Getting performance  - - - - - - - - - - - - - - - - - - -
        while self.file_result != 0:
            try:
                template_file = f"""
# This is the Python Iteration: {number_iteration}
# Author: [Your Name]
# Date: [Insert Date]

# Code:
import sys
from pathlib import Path
import ioh
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from P1 import P1

def evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas):
        
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()

    def run_metaheuristic():
        met = mh.Metaheuristic(prob, heur, num_agents, num_iterations)
        met.verbose = {self.the_bool}
        met.run()
        best_position, f_best = met.get_solution()
        return f_best, best_position

    # Ejecutar en paralelo el número de réplicas
    num_cores = min(multiprocessing.cpu_count(), num_replicas)
    results_parallel = Parallel(n_jobs=num_cores, prefer="threads")(delayed(run_metaheuristic)() for _ in range(num_replicas))

    # Extraer los valores de fitness de los resultados y calcular la métrica de rendimiento
    fitness_values = [result[0] for result in results_parallel]
    positions = [result[1] for result in results_parallel]
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    # Fitness finales
    fitness_array = np.array(fitness_values).T
    print("final_fitness_array", fitness_array)

    # Retorna el mejor valor y la mejor posición encontrada en todas las réplicas
    best_fitness_index = np.argmin(fitness_values)
    best_position = positions[best_fitness_index]
    return performance_metric, best_position, fitness_array

{self.made_metaheuristic}    

problem_id = {self.problem_id}    
instance = 1
dimension = {self.dimensions}     
num_agents= {self.num_of_agents}    
num_iterations = 100
num_replicas = 100

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)
print("Métrica de rendimiento (Mediana + IQR):", performance_metric)
print("Mejor posición encontrada:", best_position)

# Obtener y comparar con el óptimo
problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y                 
            """
                if number_iteration > 3:
                    output = ollama.generate(
                    model=self.model,
                    prompt = f"""
                        {self.prompt}

                        ### IMPORTANT INSTRUCTIONS:
                        - DO NOT PROVIDE ANY TEXT OR EXPLANATION or </think>, ONLY CODE:
                        1. **Code Format**: Avoid using triple backticks (` ``` `), Python-specific syntax, or markdown in the response.
                        2. **Operator Diversity**: Ensure extracted operators reflect a wide range of strategies. Avoid limiting to only 2-3 operator types; include variety based on the data's complexity.
                        3. **Operator Limit**: Do not include more than **4 operators** in the response.

                        ### DATA USAGE RULES:
                        - Do **not** modify operators, parameters, variables, or selectors from the provided data.
                        - Strictly adhere to the provided details without inventing, omitting, or altering any information.

                        ### OBJECTIVE:
                        Incorporate the provided feedback to design a new metaheuristic approach. Aim to create a method that maintains or improves performance while reducing computational cost.

                        ### PROVIDED DATA:
                        {data}

                        ### FEEDBACK:
                        {answer}

                        ### RESPONSE FORMAT:
                        Please structure your response using the following template:
                        heur = [
                            (  # Search Operator 1
                                '[operator_name]',
                                {{
                                    'parameter1': value1,
                                    'parameter2': value2,
                                    # Add more parameters as necessary
                                }},
                                '[selector_name]'
                            ),
                            (  # Search Operator 2
                                '[operator_name]',
                                {{
                                    'parameter1': value1,
                                    'parameter2': value2,
                                    # Add more parameters as necessary
                                }},
                                '[selector_name]'
                            )
                            # Add up to 4 operators total
                        ]
                    """
                    )
                    self.made_metaheuristic = output['response']
                    self.execute_generated_code(template_file, output_folder, number_iteration, False)
                    self.the_bool = False
                    self.execute_generated_code(template_file, output_folder, number_iteration, False)

                # Generar el resultado con el modelo
                else:
                    output = ollama.generate(
                    model=self.model,
                    prompt = f"""
                        {self.prompt}

                        ### IMPORTANT INSTRUCTIONS:
                        - DO NOT PROVIDE ANY TEXT OR EXPLANATION or </think>, ONLY CODE:
                        1. **Code Format**: Avoid using triple backticks (` ``` `), Python-specific syntax, or markdown in the response.
                        2. **Operator Diversity**: Ensure extracted operators reflect a wide range of strategies. Avoid limiting to only 2-3 operator types; include variety based on the data's complexity.
                        3. **Operator Limit**: Do not include more than **4 operators** in the response.

                        ### DATA USAGE RULES:
                        - Do **not** modify operators, parameters, variables, or selectors from the provided data.
                        - Strictly adhere to the provided details without inventing, omitting, or altering any information.

                        ### OBJECTIVE:
                        Aim to create a method that maintains or improves performance while reducing computational cost.

                        ### PROVIDED DATA:
                        {data}

                        ### RESPONSE FORMAT:
                        Please structure your response using the following template:
                        heur = [
                            (  # Search Operator 1
                                '[operator_name]',
                                {{
                                    'parameter1': value1,
                                    'parameter2': value2,
                                    # Add more parameters as necessary
                                }},
                                '[selector_name]'
                            ),
                            (  # Search Operator 2
                                '[operator_name]',
                                {{
                                    'parameter1': value1,
                                    'parameter2': value2,
                                    # Add more parameters as necessary
                                }},
                                '[selector_name]'
                            )
                            # Add up to 4 operators total
                        ]
                        """
                    )
                    self.made_metaheuristic = output['response']
                    self.execute_generated_code(template_file, output_folder, number_iteration, False)
                    self.the_bool = False
                    self.execute_generated_code(template_file, output_folder, number_iteration, False)

            except Exception as e:
                # Manejar cualquier error durante la ejecución del código generado
                print(f"Error durante la ejecución: {e}")
                print("Regenerando código...")
                continue  # Saltar al inicio del ciclo para regenerar código

        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_iteration_{number_iteration}.py"

        refined_meta_file_path = os.path.join(base_path_m, folder_name_m, file_name_m)
        input_file_path = os.path.join(output_folder, f'execution_iteration_{number_iteration}.py')


        single_performance_found, self.standard_dev = self.getting_single_metaheuristic_performance(output_folder, number_iteration)
        print("single_performance_found: ", single_performance_found)
        print(" self.standard_dev: ",  self.standard_dev)

        """  
        with open(input_file_path, 'r') as f:  
            print("hola...", self.file_contents)
            self.file_contents = f.read()
            self.extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)

        with open(input_file_path, 'r', encoding='utf-8') as file:
            metaheuristic_file = file.read()
            self.performance_found = single_performance_found
            performance_string = str(single_performance_found)
            self.feedback_collection.add(
                documents=[f"iteration number: {number_iteration}",self.extracted_metaheuristic, performance_string ],
                metadatas=[{"hnsw:space": "cosine"}, {"hnsw:space": "cosine"}, {"hnsw:space": "cosine"}],
                ids=[f"id_parameters_{number_iteration}", f"id_metaheuristic_file_{number_iteration}",  f"id_performance_found_{number_iteration}"]
            )
        """     

        if number_iteration > 0:
            self.decide_next_step(output_folder, number_iteration)
        """
        with open(input_file_path, 'r') as f:  
            print("hola...", self.file_contents)
            self.file_contents = f.read()
            self.extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)

        with open(input_file_path, 'r', encoding='utf-8') as file:
            metaheuristic_file = file.read()
            self.performance_found = single_performance_found
            performance_string = str(single_performance_found)
            self.feedback_collection.add(
                documents=[f"iteration number: {number_iteration}",self.extracted_metaheuristic, performance_string ],
                metadatas=[{"hnsw:space": "cosine"}, {"hnsw:space": "cosine"}, {"hnsw:space": "cosine"}],
                ids=[f"id_parameters_{number_iteration}", f"id_metaheuristic_file_{number_iteration}",  f"id_performance_found_{number_iteration}"]
            )

        #self.perform_optuna_tuning(output_folder, number_iteration)
        """
        return output_response  
    
    def decide_next_step(self, output_folder, number_iteration):
        std_threshold = max(9, 0.1 * self.best_performance)  
        # Umbral de desviación estándar        
        print("self.old_losses: ", self.old_losses)
        print("self.new_losses: ", self.new_losses)

        if abs(self.best_performance - self.performance_found) < 0.01:
            print("el valor", abs(self.best_performance - self.performance_found))
            print("Convergence detected. Exploiting the current best solution.")

        alpha = 0.1  # Nivel de significancia
        u_statistic, u_p_value = stats.mannwhitneyu(self.new_losses, self.old_losses, alternative='less')
        if u_p_value < alpha:
            print("Las distribucion es mejor, continua la próxima iteración.")
            print(f"U p-value: {u_p_value}")
            self.perform_optuna_tuning(output_folder, number_iteration)
        else:
            print("Las distribuciones no son significativamente diferentes. Volver a exploración.")
            self.exploration(output_folder, number_iteration)

        return std_threshold
    
    """
    Refinement:  Will enhance the metaheuristic.
    Must have the hyperparameters in order to refine it, once refined, it should not refine again, since it will be as refining the same metaheuristic twice with 
    the exact same hyperparameters and hence, performance. 
    """
   
    def refinement(self, output_folder, number_iteration):
        self.file_result = 1 # Changing again
        print("i am refinement")
        output_response_refinement = None

        # Get the metaheuristic file to REFINE IT - - - - - - - - - - - - - - - - - - -
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_iteration_{number_iteration}.py"
        meta_file_path = os.path.join(base_path_m, folder_name_m, file_name_m)
        with open(meta_file_path, 'r', encoding='utf-8') as file:
            metaheuristic_code_file = file.read()
        # Get the metaheuristic file to REFINE IT - - - - - - - - - - - - - - - - - - -

        # Refining the Metaheuristic - - - - - - - - - - - - - - - - - - -
        #  If you encounter an error, address it as follows: {self.file_result_error}.
        while self.file_result != 0:
            try:
                print("self.hyperparameters", self.hyperparameters)
                output = ollama.generate(
                    model=self.model,
                    prompt = f"""
                    {self.prompt}, taking this template: {metaheuristic_code_file}, modify it in order to put these parameters.
                    Use the following parameters for the search operators:
                    Parameters: {self.hyperparameters}, do NOT modify anything else. 

                    You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response. All outputs must be plain text. 
                    """
                )
                output_response_refinement = self.execute_generated_code(output['response'], output_folder, number_iteration, False)
            # Refining the Metaheuristic - - - - - - - - - - - - - - - - - - 
            except Exception as e:
                # Manejar cualquier error durante la ejecución del código generado
                print(f"Error durante la ejecución: {e}")
                print("Regenerando código...")
                continue  # Saltar al inicio del ciclo para regenerar código
        
        
        single_performance_found, self.standard_dev = self.getting_single_metaheuristic_performance(output_folder, number_iteration)
        # Hyperparameters to feedback - - - - - - - - - - - - - - - - - - -
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_iteration_{number_iteration}.py"

        refined_meta_file_path = os.path.join(base_path_m, folder_name_m, file_name_m)

        with open(refined_meta_file_path, 'r', encoding='utf-8') as file:
            metaheuristic_file = file.read()
        
            # Hyperparameters to feedback - - - - - - - - - - - - - - - - - - -
            if self.hyperparameters:
                # Define the expected variables
                expected_variables = {
                    'scale', 'elite_rate', 'mutation_rate', 'probability',
                    'gravity', 'alpha', 'beta', 'dt', 'mating_pool_factor',
                    'num_rands', 'pairing', 'crossover', 'radius', 'angle',
                    'sigma', 'factor', 'self_conf', 'swarm_conf', 'version',
                    'expression', 'distribution'
                }
                # Filter and process the hyperparameters
                found_hyperparameters = {
                    key: (float(value) if isinstance(value, (int, float)) else value)
                    for key, value in self.hyperparameters.items()
                    if key in expected_variables and value != ''
                }
                # Converts it into a JSON? 
                hyperparameters_str = json.dumps(found_hyperparameters, indent=4)
                # Save the string and pass it to the feedback collection
                numero = number_iteration
                #print(hyperparameters_str)  # Optional: Print the result
                print("Guardando: ", {number_iteration}, "hyperparameters_str: ", hyperparameters_str)
                self.performance_found = single_performance_found
                performance_string = str(single_performance_found)
                parameters = f"""Iteration:",  {numero}, Parameters:", {hyperparameters_str}, "Performance Found:", {single_performance_found}"""
                self.feedback_collection.add(
                    documents=[parameters, self.extracted_metaheuristic, performance_string ],
                    metadatas=[{"hnsw:space": "cosine"}, {"hnsw:space": "cosine"}, {"hnsw:space": "cosine"}],
                    ids=[f"id_parameters_{numero}", f"id_metaheuristic_file_{numero}",  f"id_performance_found_{numero}"]
                )
            else:
                print("not found: self.hyperparameters: ")
 
        return output_response_refinement
    
    
    """
    perform_optuna_tuning: 
    - Must get the hyperparameters 
    """
    def perform_optuna_tuning(self, output_folder, number_iteration):
        self.file_result = 1 # Changing again
        print("i am perform_optuna_tuning")
        # Query for the Optuna Template - - - - - - - - - - - - - - - - - - -
        optuna_template = ollama.embeddings(
        prompt="give me the optuna template",
        model=self.model_embed
        )
        results = self.optuna_collection.query(
        query_embeddings=[optuna_template["embedding"]],
        n_results=2
        )
        optuna_template = results['documents'][0][0]
        #print("optuna_template", optuna_template)
        # Query for the Optuna Template - - - - - - - - - - - - - - - - - - -

      
        checker_variable = 1 # variable to count how many times it has created a metaheuristic. 

        # Getting the Optuna Tuned Metaheuristic - - - - - - - - - - - - - - - - - - -
        input_file_path = os.path.join(output_folder, f'execution_iteration_{number_iteration}.py')
        with open(input_file_path, 'r') as f: # here is the error
            self.file_contents = f.read()
        extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)
        # Getting the Optuna Tuned Metaheuristic - - - - - - - - - - - - - - - - - - -

        # Query and Generate Optuna File - - - - - - - - - - - - - - - - - - -
        optuna_task = f"""
        You are an expert in natural computing. Your task is to generate code adhering strictly to the given template. 

        ### Rules:
        1. **Plain Text Only**: Do not include Markdown or triple backticks (```). Outputs must be plain text.
        2. **No Deviations**: Follow the provided template exactly—no additions, modifications, or extra explanations.
        3. **Use the follwoing template**: {optuna_template}
        ### Template Modifications:
        - Repace with:
        problem_id = {self.problem_id}    
        instance = 1
        dimension = {self.dimensions}     
        num_agents= {self.num_of_agents}    
        num_iterations = 100
        num_replicas = 10
        - Replace `def objective(trial):` with:
            def objective(trial):
                heur = [
                    {extracted_metaheuristic}
                ]
        - Use this format for `heur`:
            heur = [
                ('[operator_name]', {{'parameter1': value1, 'parameter2': value2}}, '[selector_name]'),
                ('[operator_name]', {{'parameter1': value1, 'parameter2': value2}}, '[selector_name]')
            ]
        Any deviation from the above instructions is incorrect.
        """

        while self.file_result != 0:
            print("checking num_agents", self.num_of_agents)
            try:
                # Generate code using the model
                output = ollama.generate(
                    model=self.model,
                    prompt=f"""{optuna_task}
                    Remember to put:     performance = evaluate_sequence_performance(heur, prob, num_agents={self.num_of_agents}, num_iterations=100, num_replicas=30)
                    """
                )
                response = output.get('response', '').strip()
                
                # Verify and execute the generated code
                if response:
                    checker_variable += 1
                    print("checker_variable--OPTUNA--->>>>>", checker_variable)
                    execution_result_optuna = self.execute_generated_code(response, output_folder, number_iteration, is_optuna=True)
                if checker_variable > 6:
                    break
            except Exception as e:
                # Handle any errors during code execution
                print(f"Error during execution: {e}")
                print("Regenerating code...")
                continue

        # Getting the Hyperparameters - - - - - - - - - - - - - - - - - - -
        current_directory = os.getcwd()
        folder_name = self.folder_name
        file_name = f"execution_optuna_result_{number_iteration}.txt"
        file_path = os.path.join(current_directory, folder_name, file_name)

        self.hyperparameters = self.get_preferential_values(file_path)
        print("numero_iteracion_optuna_tune :", number_iteration, "hyperparameters encontrados: ", self.hyperparameters)
        
        #  Important  - - - - - - - - - - - - - - - - - - -
        #Checking whether it should create again: 
        if self.hyperparameters:
            self.refinement(output_folder, number_iteration)
        else:
            #if self.performance_found > self.best_performance:
            self.exploration(output_folder, number_iteration)
        #  Important  - - - - - - - - - - - - - - - - - - -

        return execution_result_optuna


    def execute_generated_code(self, code, output_folder, number_iteration, is_optuna):
        prefix = "execution_optuna_" if is_optuna else "execution_"
        # os.path.join()`: This function is used to create a proper file path string 
        # that works across different operating systems.
        file_name =  os.path.join(output_folder, f'{prefix}iteration_{number_iteration}.py')
        with open(file_name, 'w') as f:
            f.write(code)
        try:
            result = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=140)
            execution_result = f"Exit code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            self.file_result = result.returncode
            self.file_result_error = result.stderr
            result_file_name = f'{prefix}result_{number_iteration}.txt'
            
            result_file_path = os.path.join(output_folder, result_file_name)
            with open(result_file_path, 'w') as f:
                f.write(execution_result)
                
            return execution_result
            
        except subprocess.TimeoutExpired:
            return "Execution timed out after 30 seconds"
        except Exception as e:
            return f"An error occurred during execution: {str(e)}"
        
    
    def run(self):
        self.logger.debug("Starting main execution")
        try:
            # Create output folder
            self.logger.debug("Creating output folder")
            current_dir = pathlib.Path(__file__).parent.resolve()
            output_folder_parent = current_dir / 'outputs-results'

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = output_folder_parent / f'ollama_output_{self.problem_id}_{self.dimensions}_{timestamp}'
            self.folder_name = output_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            for i in range(self.max_iterations):
                self.logger.debug(f"Starting exploration iteration {i}")
                self.exploration(output_folder, i)
            # -------------------- 
        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise         

if __name__ == "__main__":
    generator = GerateMetaheuristic(5, 5, 25)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
    