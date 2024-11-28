import ollama
import chromadb
import os
import datetime
import subprocess
import logging
import re
import pathlib
import sys
from pathlib import Path
import json
import numpy as np

#project_dir = Path(__file__).resolve().parents[2]
#sys.path.insert(0, str(project_dir))

"""
Uses: myllama3:latest 
- Creates an embedding model.
- Gets the {self.x_best} and {self.f_best} from the last iteration, in order to minimize it.
- Creates the feedback collection:

    self.feedback_collection.add(
        ids=[f"iteration_{number_iteration}"],
        embeddings=[feedback_embedding['embedding']],
        documents=[output['response'] + "\n" ],
        metadatas={"x_best": {self.x_best}, "f_best": {self.f_best}}
    )


-  Added: while self.file_result != 0 or self.f_best > self.first_f_best: 
Which means that will run till the output is correct and till the fitness is better than the previous one. 
Although there is a break after the third try. 

-- myqwen2.5:latest
-- mxbai-embed-large
"""
# will try soon: ollama pull bge-large
class GerateMetaheuristic:
    def __init__(self, benchmark_function, dimensions, max_iterations, model="myqwen2.5:latest", model_embed="all-minilm:latest"):
        self.model = model
        self.model_embed = model_embed
        self.max_iterations = max_iterations

        ## METAHEURISTIC CREATION
        self.benchmark_function = benchmark_function
        self.dimensions = dimensions
        self.hyperparameters = ""
        self.performance_found = ""
        self.best_performance = 200

        self.folder_name = ""

        ## LOGGER
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        ## ERROR CHECKER
        self.file_result = ""
        self.file_result_error = ""
        self.total_budget = 100

        ## PERFORMANCE
        self.f_best = 0
        self.first_f_best = 15

        ## FOR OPTUNA
        self.file_contents = ""

        

        ## COLLECTIONS
        client = chromadb.Client()
        self.python_collection = client.create_collection(name="python_collection")
        self.feedback_collection = client.create_collection(name="feedback_collection")
        self.optuna_collection = client.create_collection(name="optuna_collection")

        self.prompt = f"""You are a highly skilled computer scientist in the field of natural computing. Your task is to design a metaheuristic algorithm, 
        you should only use the information that was provided to you. 
        Remember that when writing the operator's names, they should be ALL in LOWER CASE AND WITH A '_' 
        instead of typing a space. Remember that, if the dimension is 3 or bigger, you should use a bigger selector, as there is more space to cover.
        Please in the 'fun' variable you must change it too: 'fun = bf.{self.benchmark_function}({self.dimensions})', do not change these values given. 
        """""

        python_files_directory = os.path.join(os.path.dirname(__file__), 'metaheuristic_builder')
        optuna_files_dir = os.path.join(os.path.dirname(__file__), 'optuna_builder')
        for d in os.listdir(python_files_directory):
            file_path = os.path.join(python_files_directory, d)
            if os.path.isfile(file_path):  # Check if it's a file
                file_content = self.read_file(file_path)
                response = ollama.embeddings(model=self.model_embed, prompt=file_content)
                embedding = response.get("embedding")
                if embedding:
                    self.python_collection.add(
                        ids=[d],
                        embeddings=[embedding],
                        documents=[file_content],
                        metadatas=[{"filename": d}]
                    )
                    print(f"Added {d} to the collection")
                else:
                    print(f"Warning: Empty embedding generated for {d}")
            else:  # If it's not a file, skip it
                continue
                
        """  
        for d in os.listdir(optuna_files_dir):
            file_path = os.path.join(python_files_directory, d)
            if os.path.isfile(file_path):  # Check if it's a file
                file_content = self.read_file(file_path)
                response = ollama.embeddings(model=self.model_embed, prompt=file_content)
                embedding = response.get("embedding")
                if embedding:
                    self.optuna_collection.add(
                        ids=[d],
                        embeddings=[embedding],
                        documents=[file_content],
                        metadatas=[{"filename": d}]
                    )
                    print(f"Added {d} to the collection")
                else:
                    print(f"Warning: Empty embedding generated for {d}")
            else:  # If it's not a file, skip it
                continue     
        """
    def extract_code_from_code_with_optuna(self, code_file):
        response_optuna = ollama.embeddings(
        prompt="See all the operators, with its parameters and selectors provided",
        model=self.model_embed
        )
        results_optuna = self.python_collection.query(
        query_embeddings=[response_optuna["embedding"]],
        n_results=1
        )
        optuna_data = results_optuna['documents'][0][0]

        pattern = r'heur\s*=\s*\[(.*?)\]' 
        match = re.search(pattern, code_file, re.DOTALL)
 
        if match:
            extracted_content = match.group(1).strip()  
            print("MATCH", extracted_content)

            output = ollama.generate(
            model = self.model,
            prompt = f"""Modify this metaheuristic: {extracted_content}. 
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
                            'name_variable': trial.suggest_float('name_variable', 0.01, 0.9)
                        - Always include a comma after the modified parameter.

                        2. **If the parameter provides a category:**
                        - Modify it to the format:
                            'category_name': trial.suggest_categorical('category_name', ['option_1', 'option_2', 'option_3'])
                        - Include at least **two options**. Avoid using only one option.
                        - **Incorrect Example:**
                            'category_name': trial.suggest_categorical('category_name', ['option_1'])
                        - **Correct Example:**
                            'category_name': trial.suggest_categorical('category_name', ['option_1', 'option_2', 'option_3'])

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
    
    def process_file(self, filepath):
        f_best_values = []  # Collect all f_best values in the file
        base_path_m = "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results"
        folder_name_m = self.folder_name
        my_file_path = os.path.join(base_path_m, folder_name_m, filepath)


        with open(my_file_path, 'r') as file:
            for line in file:
                # Search for f_best in the current line
                matches = re.findall(r'f_best = ([0-9\.]+)', line)
                
                # If a match is found, store it in the list
                if matches:
                    f_best_values.append(float(matches[0]))
                print(f_best_values)
        return f_best_values

    # Function to calculate performance metrics
    def calculate_performance(self, f_best_values_list):   
        performances = []
        for f_best_values in f_best_values_list:
            if f_best_values:  # Check if the list is not empty
                med = np.median(f_best_values)
                iqr = np.percentile(f_best_values, 75) - np.percentile(f_best_values, 25)
                performance_metric = med + iqr
                performances.append(performance_metric)
            else:
                performances.append(None)  # Placeholder for missing data
        
        # Convert the list of performances to a formatted string
        performance_string = ", ".join(
            f"{p:.2f}" if p is not None else "None" for p in performances
        )
        return performance_string

    # USING THE OPTUNA BEST PARAMETERS NOW:
    def get_preferential_values(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code_file = file.read()
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None, None

        # Regex patterns
        hyperparameters_pattern = r"Mejores hiperparámetros encontrados:\n({.*?})"
        performance_pattern = r"Mejor rendimiento encontrado:\n([\d.]+)"

        hyperparameters_match = re.search(hyperparameters_pattern, code_file, re.DOTALL)
        hyperparameters_dict = eval(hyperparameters_match.group(1)) if hyperparameters_match else None

        performance_match = re.search(performance_pattern, code_file)
        performance_found = float(performance_match.group(1)) if performance_match else None

        return hyperparameters_dict, performance_found

        
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
        
    def self_refine(self, output_folder, number_iteration):
        #current_output = ""
        relevant_feedback = ""
        checker_variable = 1 # variable to count how many times it has created a metaheuristic. 
        
        # QUERY FOR THE OPERATORS 
        output = ollama.embeddings(
        prompt=f"give me the best operators, operators' parameters and selectors for 'fun = bf.{self.benchmark_function}({self.dimensions})'",
        model=self.model_embed
        )
        results = self.python_collection.query(
        query_embeddings=[output["embedding"]],
        n_results=3
        )
        data = results['documents'][0][0]
        #print("data-from-py-collection----", data)
        # QUERY FOR THE OPERATORS 
        
        # QUERY FOR THE TEMPLATE 
        output_template = ollama.embeddings(
        prompt="give me the given template to create a metaheuristic properly",
        model=self.model_embed
        )
        results = self.python_collection.query(
        query_embeddings=[output_template["embedding"]],
        n_results=2
        )
        data_template = results['documents'][0][0]
        print("data-from-py-collection----", data)
        # QUERY FOR THE TEMPLATE 

        ## PERFORMANCE BUT WHILE CHECKING IF FILE EXISTS:
        num = number_iteration - 1
        file_name = f"execution_result_{num}.txt"

        # Check if the file exists
        if os.path.exists(file_name):
            # Process the file if it exists
            all_f_values = self.process_file(file_name)
            last_performance = self.calculate_performance(all_f_values)
            try:
                last_performance_list = [float(value.strip()) for value in last_performance.split(",")]
                last_performance_int = sum(last_performance_list) / len(last_performance_list)
                print("Best performance so far, in while:", last_performance_int)
            except ValueError as e:
                print("Error processing performance values:", e)
        else:
            # Continue to the next iteration if the file does not exist
            print(f"File {file_name} does not exist, skipping...")
        ## PERFORMANCE BUT WHILE CHECKING IF FILE EXISTS:


         ## OPTUNA CHECKER
        if number_iteration > 1 or number_iteration == 1:  # if something changes
            ## Getting the performance execution_result_0.txt
            all_f_values = self.process_file(f"execution_result_{number_iteration-1}.txt")
            last_performance = self.calculate_performance(all_f_values)
            # Split the string by commas and strip whitespace
            last_performance_list = [float(value.strip()) for value in last_performance.split(",")]
            last_performance_int = sum(last_performance_list) / len(last_performance_list)
            print("Best performance so far, above 1 iter:",last_performance_int )

            #metaheuristic_file = f"/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/{self.folder_name}/execution_iteration_{number_iteration-1}.py"
            base_path_m = "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results"
            folder_name_m = self.folder_name
            file_name_m = f"execution_iteration_{number_iteration-1}.py"

            meta_file_path = os.path.join(base_path_m, folder_name_m, file_name_m)

            with open(meta_file_path, 'r', encoding='utf-8') as file:
                metaheuristic_code_file = file.read()

            base_path = "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results"
            folder_name = self.folder_name
            file_name = f"execution_optuna_result_{number_iteration-1}.txt"

            # Construct the file path
            file_path = os.path.join(base_path, folder_name, file_name)
            print("Constructed file path:", file_path)

            # Check if the file exists
            if not os.path.exists(file_path):
                print("Error: File does not exist:", file_path)
            else:
                # Ensure the file is readable
                if not os.access(file_path, os.R_OK):
                    print("Error: File is not readable:", file_path)
                else:
                    print("File exists and is readable. Proceeding...")
                    # Call the function
                    self.hyperparameters, self.performance_found = self.get_preferential_values(file_path)
                    print("Extracted Hyperparameters:", self.hyperparameters)
                    print("Extracted Performance:", self.performance_found)
            
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

                parameters_str = json.dumps(found_hyperparameters, indent=4)

                # Save the string and pass it to the feedback collection
                numero = number_iteration
                print(parameters_str)  # Optional: Print the result

                self.feedback_collection.add(
                    documents=[metaheuristic_code_file, parameters_str, last_performance],
                    metadatas=[{'source': 'metaheuristic_code_file'}, {'source': 'parameters'}, {'source': 'last_performance'}],
                    ids=[f"{numero}_id_metaheuristic", f"{numero}_id_optuna_parameters", f"{numero}_id_last_performance"]
                )

                
        query_embedding = ollama.embeddings(model=self.model_embed, prompt="give me the metaheuristics with the best performance")
        # Set the number of feedback results to retrieve
        n_results = max(1, min(number_iteration, 7))
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        print("relevant,feedback", relevant_feedback)
        #print(f"Retrieved data: {data}")
        #print("Data being used:", data)
        print("Prompt being passed:", f"Using this data: ...  Respond to this prompt: {self.prompt}, these are the hy: {self.hyperparameters}")
        #print(f"These are ---- {self.feedback_collection.peek()}")
        # generate a response combining the prompt and data we retrieved in step 2
        output = ollama.generate(
        model = self.model,
        prompt=f"""
        Respond to this prompt: {self.prompt}. 
        From the data provided, extract operators while ensuring a variety of strategies. Do not limit to just two or three specific types.
        Strictly match the information below, without inventing or modifying any details:
        Data, (DO NOT MODIFY ANY GIVEN OPERATORS, PARAMETERS, VARIABLES OR SELECTORS):
        {data}
        
        Use the following parameters for the search operators:
        Parameters: {self.hyperparameters} (if none are provided, you may skip this part).
                

        Use the following template for your response:
        {data_template}

        If you encounter an error, address it as follows: {self.file_result_error}.
        If any part of the data is incomplete or unavailable, state explicitly: "Information not available."
        """

        #temperature=0.7
        ) 
        self.execute_generated_code(output['response'], output_folder, number_iteration, False)
        #print("execution_result-need-to-see", execution_results

        # Must create a functionable and better-fitness metaheuristic

        # Not using the feedback, because it should only retrieve better feedback. 
        """  
        # Process feedback using the response
        response_text = output['response']
        query_embedding = ollama.embeddings(model=self.model_embed, prompt=response_text)

        # Set the number of feedback results to retrieve
        n_results = max(1, min(number_iteration, 7))
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        """

        while self.file_result != 0 or last_performance_int > self.best_performance:
            # Debugging data and feedback
            #print(f"Loop Iteration {checker_variable}, Data: {data}, Feedback: {relevant_feedback}")

            output = ollama.generate(
                model=self.model,
                prompt = f"""
                You can take into account the following feedback:
                {relevant_feedback}
                Respond to this prompt: {self.prompt},
                From the data provided, extract operators while ensuring a variety of strategies. Do not limit to just two or three specific types.
                Strictly match the information below, and you need to use exactly these parameters {self.hyperparameters} (if no parameters given, ignore it).
                Do not invent information.
                {data}
                
                Use the following template for your response:
                {data_template}

                If you encounter an error, address it as follows: {self.file_result_error}.
                If any part of the data is incomplete or unavailable, state explicitly: "Information not available."
                """
            )
            response = output.get('response', "")
            if not response:
                print("No valid response generated. Skipping iteration.")
                continue
            try:
                self.execute_generated_code(response, output_folder, number_iteration, False)
            except Exception as e:
                print(f"Error executing code: {e}")
                continue

            if checker_variable > 6:
                print("More than 6 iterations so far ...")

            if number_iteration > 1 or number_iteration == 1: 
                num = number_iteration-1
                all_f_values = self.process_file(f"execution_result_{num}.txt")
                last_performance = self.calculate_performance(all_f_values)
                last_performance_list = [float(value.strip()) for value in last_performance.split(",")]
                last_performance_int = sum(last_performance_list) / len(last_performance_list)
                print("Best performance so far, in while:",last_performance_int )
            else:
                last_performance_int = self.best_performance
                print("Best performance so far:",last_performance_int )


                
        if last_performance_int < self.best_performance:
            self.best_performance = last_performance_int
     
        output_response = output['response']

        """  
        # Ensuring that the variable chunk is always a list.
        if not isinstance(output_response, list): 
            chunks = [output_response]   
        else:
            chunks = output_response    

        filename = f"execution_result_{number_iteration}.txt"
        modelname = "mxbai-embed-large"

        # Retrieve or compute embeddings and save metadata
        embeddings, metadata = self.get_embeddings(filename, modelname, chunks, output_response, output['response'])
        print("embeddings",embeddings)
        print("metadata",metadata)
        """

        """   
        feedback_embedding = ollama.embeddings(model=self.model_embed, prompt=output['response'])
        self.feedback_collection.add(
            ids=[f"iteration_{number_iteration}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[output['response'] + "\n" ],
            metadatas={"f_best": self.f_best}
        )
        
        query_embedding = ollama.embeddings(model=self.model_embed, prompt=output['response'])
        n_results = 1
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        #print("relevant_feedback", relevant_feedback)
        #print("relevant_feedback['documents']}", relevant_feedback['documents'])
        """ 

        return output_response


    def self_refine_with_optuna(self, output_folder, number_iteration):
        # Here the feedback is not provided, but saved for the next metaheuristics creations. 

        current_output = ""
        checker_variable = 1 # variable to count how many times it has created a metaheuristic. 
        input_file_path = os.path.join(output_folder, f'execution_iteration_{number_iteration}.py')
        with open(input_file_path, 'r') as f:
            self.file_contents = f.read()
        extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)


        optuna_task =  f"""You are a highly skilled computer scientist in the field of natural computing. 
        Your task is to make the optuna algorithm of the given metaheuristic. 
        You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response.
        All outputs must be plain text. 
        You must write exactly the following code, DO NOT INVENT NEW THINGS, NOR WORDS, NOR PARAGRAPHS, NOR ANYTHING, FOLLOW THE EXACT TEMPLATE:

import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def evaluate_sequence_performance(sequence, prob, num_agents, num_iterations, num_replicas):
    def run_metaheuristic():
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        _, f_best = met.get_solution()
        return f_best

    num_cores = multiprocessing.cpu_count()
    results_parallel = Parallel(n_jobs=num_cores)(delayed(run_metaheuristic)() for _ in range(num_replicas))

    fitness_values = results_parallel
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    return performance_metric

def objective(trial):
    heur = [
        {extracted_metaheuristic} 
    ]

    fun = bf.{self.benchmark_function}({self.dimensions})
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)

    return performance

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)
"""""

        current_output_optuna = ollama.generate(
            model = self.model,
            prompt= f"""Follow this prompt: {optuna_task}
            """
        )

        execution_result_optuna = self.execute_generated_code(current_output_optuna['response'], output_folder, number_iteration, True)
        # execution_result_optuna will be the error, or trials. Must save it on the feeback if the output ran well (0)
        print("FIRST----execution_result_optuna--NEEDTOSE", execution_result_optuna)

        #or (self.f_best > self.first_f_best)
        while self.file_result != 0: 
            # generate a response combining the prompt and data we retrieved in step 2
            extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)
            output = ollama.generate(
            model = self.model,
            prompt=f"""Respond to this prompt: {optuna_task},
            Remeber that when writing the response You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response.
            If you encounter an error, address it as follows: {self.file_result_error}.
            If any part of the data is incomplete or unavailable, state explicitly: "Information not available.
            """
            ) 
            #  but fix the given error, which is: {self.file_result_error},
            if print(output['response']) != "":
                checker_variable += 1
            print("checker_variable--OPTUNA--->>>>>", checker_variable)

            execution_result_optuna = self.execute_generated_code(output['response'], output_folder, number_iteration, True)
            #print("execution_result_optuna--NEEDTOSE", execution_result_optuna)
            current_output_optuna = output
            #print("current_output_optuna-need-to-se", current_output_optuna)
            if checker_variable >= 3:
                print("Reached maximum iterations, exiting loop.")
                
        
        
        print("current_output_optuna-need-to-see-outside-while", current_output_optuna)
        
        """ 
        ## FEEDBACK PROCESS: Must be after the while, since I must only store the metaheuristics that ran well. 
        feedback_embedding = ollama.embeddings(model=self.model_embed, prompt=current_output_optuna['response'])
        self.feedback_collection.add(
            ids=[f"iteration_{number_iteration}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[current_output_optuna['response'] + "\n" ]
        )
        
        query_embedding = ollama.embeddings(model=self.model_embed, prompt=current_output_optuna['response'])
        n_results = 1
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        print("relevant_feedback", relevant_feedback)
        print("relevant_feedback['documents']}", relevant_feedback['documents'])
        """

        return current_output



    def extract_best_performance(self, output_file):
        with open(output_file, "r") as f:
            content = f.readlines()
        for line in reversed(content):
            line = line.strip()
            if 'x_best =' in line and 'f_best =' in line:
                f_best_str = line.split('f_best =')[1].split()[0].replace(',', '')
                self.f_best = float(f_best_str)
                
                return self.f_best
        print("No valid line with 'f_best' found in the file.")
        return None



    def execute_generated_code(self, code, output_folder, number_iteration, is_optuna):
        prefix = "execution_optuna_" if is_optuna else "execution_"
        # os.path.join()`: This function is used to create a proper file path string 
        # that works across different operating systems.
        file_name =  os.path.join(output_folder, f'{prefix}iteration_{number_iteration}.py')
        with open(file_name, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=200)
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
        #self.client = chromadb.Client()
        self.logger.debug("Starting main execution")
        #self.client.delete_collection(name="metaheuristic_builder")
        #self.client.delete_collection(name="optuna_collection")
        #self.client.delete_collection(name="feedback_collection")
        try:
            # Create output folder
            self.logger.debug("Creating output folder")
            current_dir = pathlib.Path(__file__).parent.resolve()
            output_folder_parent = current_dir / 'outputs-results'

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = output_folder_parent / f'ollama_output_{self.benchmark_function}({self.dimensions})_{timestamp}'
            self.folder_name = output_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            for i in range(self.max_iterations):
                self.logger.debug(f"Starting refinement iteration {i}")
                self.self_refine(output_folder, i)
                # self.logger.info(f"Refined output for iteration {i} generated")
                self.self_refine_with_optuna(output_folder, i)

            # -------------------- 
        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise         

if __name__ == "__main__":
    generator = GerateMetaheuristic("Rastrigin", 13, 15)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
    