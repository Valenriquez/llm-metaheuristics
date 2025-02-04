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

"""
Uses: ollama model: "phi4"
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
    def __init__(self, benchmark_function, dimensions, max_iterations, model="qwen2.5-coder:latest", model_embed="all-minilm:latest"):
        
        self.model = model
        self.model_embed = model_embed
        self.max_iterations = max_iterations

        ## METAHEURISTIC CREATION
        self.benchmark_function = benchmark_function
        self.dimensions = dimensions
        self.hyperparameters = ""
        self.performance_found = 200.00
        self.best_performance = 190.00

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
        self.optuna_collection = chroma_client.create_collection(name="optuna_collection", embedding_function=google_ef)

        self.creating_collection("operators_collection", self.operators_collection)
        self.creating_collection("metaheuristic_template_collection", self.metaheuristic_template_collection)
        self.creating_collection("optuna_builder", self.optuna_collection)

        self.prompt = f"""You are a highly skilled computer scientist in the field of natural computing. 
        Your task is to design a metaheuristic algorithm but you should only use the information that was provided to you to create the metaheuristic. 
        Remember that when writing the operator's names, they should be ALL in  lower case and with a '_' instead of typing a space. 
        Remember that the metaheuristic you will be creating is for: fun = bf.{self.benchmark_function}({self.dimensions} , where the number inside the '()' 
        are the dimensions. For that reason, in the 'fun' variable you must change it to: 'fun = bf.{self.benchmark_function}({self.dimensions})', do not change these values given. 
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
    
    """
    getting_single_metaheuristic_performance:  Gets performance from metaheuristic fitness
    """
    def getting_single_metaheuristic_performance(self, output_folder, number_iteration):
        # Leer archivo de resultados
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_result_{number_iteration}.txt"
        execution_path = os.path.join(base_path_m, folder_name_m, file_name_m)

        with open(execution_path, 'r') as file:
            content = file.read()

        # Validar y extraer contenido entre corchetes
        if '[' in content and ']' in content:
            start = content.index('[')
            end = content.index(']')
            array_str = content[start+1:end]

            # Convertir la cadena en un arreglo NumPy
            try:
                final_fitness = np.fromstring(array_str, sep=' ')
                if final_fitness.size == 0:
                    raise ValueError("El arreglo está vacío.")
                
                # Calcular rendimiento
                self.performance_found = self.calculate_performance(final_fitness)
                print("Getting: performance_final_fitness", self.performance_found)
            except ValueError as e:
                print(f"Error al procesar array_str: {e}")
                self.performance_found = None
        else:
            print("Error: No se encontraron corchetes en el contenido del archivo.")
            self.performance_found = None

        # Verificar si el rendimiento es válido
        if self.performance_found is not None:
            return self.performance_found

    """
    get_metaheuristic_performance:  Gets performance from metaheuristic fitness
    """
    def get_metaheuristic_performance(self, output_folder, number_iteration):
        # Leer archivo de resultados
        print("get_metaheuristic_performance activated")
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_result_{number_iteration}.txt"
        execution_path = os.path.join(base_path_m, folder_name_m, file_name_m)

        with open(execution_path, 'r') as file:
            content = file.read()

        # Validar y extraer contenido entre corchetes
        if '[' in content and ']' in content:
            start = content.index('[')
            end = content.index(']')
            array_str = content[start+1:end]

            # Convertir la cadena en un arreglo NumPy
            try:
                final_fitness = np.fromstring(array_str, sep=' ')
                if final_fitness.size == 0:
                    raise ValueError("El arreglo está vacío.")
                
                # Calcular rendimiento
                self.performance_found = self.calculate_performance(final_fitness)
                print("Getting: performance_final_fitness", self.performance_found)
            except ValueError as e:
                print(f"Error al procesar array_str: {e}")
                self.performance_found = None
        else:
            print("Error: No se encontraron corchetes en el contenido del archivo.")
            self.performance_found = None

        # Verificar si el rendimiento es válido
        if self.performance_found is not None:
            if self.performance_found < self.best_performance:
                # Actualizar el mejor rendimiento y continuar
                print("Iteracion", {number_iteration}, "self.performance_found:", self.performance_found, "self.best_performance:", self.best_performance)
                self.best_performance = self.performance_found
                print("Nuevo mejor rendimiento encontrado:", self.best_performance)
                return self.performance_found
            elif self.performance_found > self.best_performance:
                # Volver a generar código y reevaluar
                print("El rendimiento actual no es mejor. Regenerando código...")
                self.exploration(output_folder, number_iteration)
                return self.performance_found
      

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
        


        # Query for the Operators - - - - - - - - - - - - - - - - - - -
        
        # Query for the Template - - - - - - - - - - - - - - - - - - -
        output_template = ollama.embeddings(
        prompt="Give me the metaheuristic template",
        model=self.model_embed
        )
        results = self.metaheuristic_template_collection.query(
        query_embeddings=[output_template["embedding"]],
        n_results=1
        )
        data_template = results['documents'][0][0]
        print("this is metaheuristic template ", data_template)
        # Query for the Template - - - - - - - - - - - - - - - - - - -


        # there won´t be any feedback else:
        if number_iteration > 0:
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

            """ 
            # Query for the Operators - - - - - - - - - - - - - - - - - - -
            output = ollama.embeddings(
            prompt=f"give me the best operators, operators that are NOT these: {data_feedback}",
            model=self.model_embed
            )
            results = self.python_collection.query(
            query_embeddings=[output["embedding"]],
            n_results=3
            )
            data = results['documents'][0][0]
            # Query for the Operators - - - - - - - - - - - - - - - - - - -
            """

       # Main Prompt - - - - - - - - - - - - - - - - - - -
        # and (performance_final_fitness < self.best_performance)
        # Getting performance  - - - - - - - - - - - - - - - - - - -
        """ 
        if number_iteration > 2:
            complete_feedback = self.feedback_collection.get(
            include=["documents"]
            )

            output = ollama.generate(
                model=self.model,
                prompt=f"
                - Find the smallest `performance_found` from this feedback: {complete_feedback} as a baseline.
                - Maintain the operators and parameters from the provided data.
                - Add convergence-enhancing strategies to your metaheuristic design, without modifying the template.
                Use this template for your response:
                {data_template}
                "
            )
            answer = output['response']
            print("answer: feedback: ", answer)
        """ 
        
        while self.file_result != 0:
            try:
                print("Inicio de while: self.best_performance: ", self.best_performance)
                if number_iteration > 2:
                        output = ollama.generate(
                        model=self.model,
                        prompt=f"""
                        {self.prompt} 
                        DO NOT USE ```python and do NOT use any markdown code or use the triple backticks  (```) anywhere in your response.
                        From the data provided, extract operators while ensuring a variety of strategies. 
                        Do not limit to just two or three specific types.
                        
                        Use this feedback to get inspired to create a new metaheuristics with a smaller performance.
                        Feedback : {answer}.
                        Strictly match the information below, (without inventing or modifying any data, unless its the operators):
                        {data}

                        Use the following template for your response:
                        {data_template}

                        Remember to put:     met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents={self.num_of_agents})  
                        If you encounter an error, address it as follows: {self.file_result_error}.
                        """
                    )
                # Generar el resultado con el modelo
                else:
                    output = ollama.generate(
                        model=self.model,
                        prompt=f"""
                        {self.prompt} 
                        DO NOT USE ```python and do NOT use any markdown code or use the triple backticks  (```) anywhere in your response.
                        From the data provided, extract operators while ensuring a variety of strategies. 
                        Do not limit to just two specific types.
                        Strictly match the information below, without inventing or modifying any details:
                        {data}

                        Do not modify the information given.
                        Use the following template for your response:
                        {data_template}
                        Remember to put: met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents={self.num_of_agents})  
                        If you encounter an error, address it as follows: {self.file_result_error}.
                        """
                    )
                output_response = self.execute_generated_code(output['response'], output_folder, number_iteration, False)
            except Exception as e:
                # Manejar cualquier error durante la ejecución del código generado
                print(f"Error durante la ejecución: {e}")
                print("Regenerando código...")
                continue  # Saltar al inicio del ciclo para regenerar código
        
        
      
        self.perform_optuna_tuning(output_folder, number_iteration)

          

        # Calling optuna - - - - - - - - - - - - - - - - - - -
        #self.perform_optuna_tuning(output_folder, number_iteration)

        # Veryfying if exploration or refinement - - - - - - - - - - - - - - - - - - -


        """ 
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_iteration_0.py"

        meta_file_path = os.path.join(base_path_m, folder_name_m, file_name_m)
        if  os.path.exists(meta_file_path):
            print("yes it does exists")
       """
            
        return output_response  ## N.S

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
        
        
        single_performance_found = self.getting_single_metaheuristic_performance(output_folder, number_iteration)
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
                    for key, value in self.hyperparameters.items()  # Use .items() for dictionaries
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
        # Hyperparameters to feedback - - - - - - - - - - - - - - - - - - -
        
        #if number_iteration > self.refinement_starter:
        #    self.get_metaheuristic_performance(output_folder, number_iteration)
 
        
        
        # Performance - - - - - - - - - - - - - - - - - - -
        #if self.performance_found < self.best_performance:
        #    self.best_performance = self.performance_found
        #else:
        #    print("El rendimiento no fue bueno, vamos a exploration")
        #    self.exploration(output_folder, number_iteration)
        # Performance - - - - - - - - - - - - - - - - - - -

 
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
        prompt="Optuna template",
        model=self.model_embed
        )
        results = self.optuna_collection.query(
        query_embeddings=[optuna_template["embedding"]],
        n_results=1
        )
        optuna_template = results['documents'][0][0]
        #print("optuna_template", optuna_template)
        # Query for the Optuna Template - - - - - - - - - - - - - - - - - - -

      
        checker_variable = 1 # variable to count how many times it has created a metaheuristic. 

        # Getting the Optuna Tuned Metaheuristic - - - - - - - - - - - - - - - - - - -
        input_file_path = os.path.join(output_folder, f'execution_iteration_{number_iteration}.py')
        with open(input_file_path, 'r') as f: # here is the error
            self.file_contents = f.read()
        self.extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)
        # Getting the Optuna Tuned Metaheuristic - - - - - - - - - - - - - - - - - - -

        # Query and Generate Optuna File - - - - - - - - - - - - - - - - - - -
        optuna_task = f"""
        You are an expert in natural computing. Your task is to generate code adhering strictly to the given template. 

        ### Rules:
        1. **Plain Text Only**: Do not include Markdown or triple backticks (```). Outputs must be plain text.
        2. **No Deviations**: Follow the provided template exactly—no additions, modifications, or extra explanations.
        3. **Use the follwoing template**: {optuna_template}
        ### Template Modifications:
        - Replace `def objective(trial):` with:
            def objective(trial):
                heur = [
                    {self.extracted_metaheuristic}
                ]
        - Use this format for `heur`:
            heur = [
                ('[operator_name]', {{'parameter1': value1, 'parameter2': value2}}, '[selector_name]'),
                ('[operator_name]', {{'parameter1': value1, 'parameter2': value2}}, '[selector_name]')
            ]
        - Set `fun` as:
            fun = bf.{self.benchmark_function}({self.dimensions})

        Any deviation from the above instructions is incorrect.
        """

        while self.file_result != 0:
            print("checking num_agents", self.num_of_agents)
            try:
                # Generate code using the model
                output = ollama.generate(
                    model=self.model,
                    prompt=f"""{optuna_task}
                    Remember to put:     
                    performance = evaluate_sequence_performance(heur, prob, num_agents={self.num_of_agents}, num_iterations=100, num_replicas=30)
                    """
                )
                response = output.get('response', '').strip()
                
                # Verify and execute the generated code
                if response:
                    checker_variable += 1
                    print("checker_variable--OPTUNA--->>>>>", checker_variable)
                    execution_result_optuna = self.execute_generated_code(response, output_folder, number_iteration, is_optuna=True)
                if checker_variable == self.break_tries :
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
            result = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=68)
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
            output_folder = output_folder_parent / f'ollama_output_{self.benchmark_function}_{self.dimensions}_{timestamp}'
            self.folder_name = output_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            for i in range(self.max_iterations):
                if self.performance_found <= 0:
                    break
                self.logger.debug(f"Starting exploration iteration {i}")
                self.exploration(output_folder, i)
                #self.refinement(output_folder, i)
                # self.logger.info(f"Refined output for iteration {i} generated")
                # self.self_refine_with_optuna(output_folder, i)

            # -------------------- 
        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise         

if __name__ == "__main__":
    generator = GerateMetaheuristic("Bohachevsky", 3, 15)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
    0