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

-- qwen2.5:latest
-- mxbai-embed-large
- llama3.3
- 
"""
# will try soon: ollama pull bge-large
class GerateMetaheuristic:
    def __init__(self, benchmark_function, dimensions, max_iterations, model="qwen2.5-coder:latest", model_embed="all-minilm:latest"):
        self.model = model
        self.model_embed = model_embed
        self.max_iterations = max_iterations

        ## METAHEURISTIC CREATION
        self.benchmark_function = benchmark_function
        self.dimensions = dimensions
        self.hyperparameters = ""
        self.performance_found = 0.00
        self.best_performance = 200.00

        self.folder_name = ""

        ## LOGGER
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        ## ERROR CHECKER
        self.file_result = 1
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
        instead of typing a space. 
        Please in the 'fun' variable you must change it too: 'fun = bf.{self.benchmark_function}({self.dimensions})', do not change these values given. 
        """""

        python_files_directory = os.path.join(os.path.dirname(__file__), 'metaheuristic_builder')
        #optuna_files_dir = os.path.join(os.path.dirname(__file__), 'optuna_builder')
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


    """
    get_preferential_values:  Gets the best parameters and performance from optuna 

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
        performance_pattern = r"Mejor rendimiento encontrado:\n([\d.]+)"

        hyperparameters_match = re.search(hyperparameters_pattern, code_file, re.DOTALL)
        hyperparameters_dict = eval(hyperparameters_match.group(1)) if hyperparameters_match else None

        performance_match = re.search(performance_pattern, code_file)
        performance_found = float(performance_match.group(1)) if performance_match else None
        print("performance_found ---->",performance_found)

        return hyperparameters_dict, performance_found

        
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
        
    """
    Exploration:  Will create a metaheuristic and search for new creations. After the first iteration it will look on the feedback for inspiration on better 
    metaheuirsics creation. 

    - Will have access to feedback
    - Will get hyperparameters for the refinement 
    """
    def exploration(self, output_folder, number_iteration):
        print("i am exploration")
        print("Beginning with exploration number----", number_iteration)
        data_feedback = ''
        output_response = None
        # Query for the Operators - - - - - - - - - - - - - - - - - - -
        output = ollama.embeddings(
        prompt=f"give me the best operators, operators' parameters and selectors for 'fun = bf.{self.benchmark_function}({self.dimensions})'",
        model=self.model_embed
        )
        results = self.python_collection.query(
        query_embeddings=[output["embedding"]],
        n_results=3
        )
        data = results['documents'][0][0]
        # Query for the Operators - - - - - - - - - - - - - - - - - - -
        
        # Query for the Template - - - - - - - - - - - - - - - - - - -
        output_template = ollama.embeddings(
        prompt="give me the given template to create a metaheuristic properly",
        model=self.model_embed
        )
        results = self.python_collection.query(
        query_embeddings=[output_template["embedding"]],
        n_results=2
        )
        data_template = results['documents'][0][0]
        # Query for the Template - - - - - - - - - - - - - - - - - - -

        # there won´t be any feedback else:
        if number_iteration >= 1:
            # Query for the Feedback - - - - - - - - (asks for the metaheuristics and feedback) - - - - - - - - - - -
            output_feedback = ollama.embeddings(
            prompt="give me all the metaheuristics created with their given performance",
            model=self.model_embed
            )
            results = self.feedback_collection.query(
            query_embeddings=[output_feedback["embedding"]],
            n_results=1
            )
            data_feedback = results['documents'][0]
            print("data_feedback: give me all the metaheuristics created with their given performance", data_feedback)
            # Query for the Feedback - - - - - - - - - - - - - - - - - - -

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

        matches = 0
        metaheuristics_search_terms = set([
            "random_search", "central_force_dynamic", "differential_mutation",
            "firefly_dynamic", "genetic_crossover", "genetic_mutation",
            "gravitational_search", "random_flight", "local_random_walk",
            "random_sample", "spiral_dynamic", "swarm_dynamic"
        ])

        # Main Prompt - - - - - - - - - - - - - - - - - - -
        while not matches: # It needs to have a correct output
            # First Prompt - - - - - - - - - - - - - - - - - - -
            print("No matches found. Retrying...")
            output = ollama.generate(
            model = self.model,
            prompt=f"""
           You are a highly skilled computer scientist and an expert in natural computing, specializing in designing innovative metaheuristic algorithms. Your objective is to create a metaheuristic design inspired by the provided data and feedback. Follow these precise guidelines:
            1. **Operator Naming Conventions**:  
            All operator names must be in **lowercase** and use **underscores** (`_`) to separate words (e.g., `example_operator_name`).

            2. **Feedback Utilization**:  
            - If feedback is provided (`{data_feedback}`), use it as inspiration to guide your design.  
            - **Do not copy** the exact metaheuristic described in the feedback.  
            - **Do not invent entirely new operators or selectors**; instead, adapt and innovate based on the strategies and patterns observed in the feedback.  
            - If no feedback is provided, you may skip this part entirely.

            3. **Data-Based Design**:  
            - Use the provided data (`{data}`) to extract and incorporate operators with their selectors. Do not add any operator that is not in that 'data'.  
            - Ensure a diverse and balanced selection of strategies, avoiding over-reliance on just two or three types.  
            - Strictly adhere to the details in the data—do not invent, modify, or extrapolate beyond what is provided.

            4. **Design Philosophy**:  
            - The resulting metaheuristic should embody creativity and diversity while staying grounded in the provided data and feedback.  
            - Avoid redundancy and ensure clarity in the structure and functionality of the operators.

            Deliver a detailed metaheuristic design, adhering to these constraints and conventions.
            This is the format that you need to use (PLEASE FOLLOW STRICTYLY THE FOLLOWING FORMAT, DO NOT INVENT NEW CODE, DO NOT INVENT PARAMETERS, NOR OPERATORS, NOR SELECTORS NEITHER 
            AND DO NOT USE MORE THAN FOUR (4) METAHEURISTICS):    
                heur = [
                    (  # Search operator 1
                        '[operator_name]',
                        {{
                            'parameter1': value1,
                            'parameter2': value2,
                            more parameters as needed
                        }},
                        '[selector_name]'
                    ),
                    (
                        '[operator_name]',
                        {{
                            'parameter1': value1,
                            'parameter2': value2,
                            ... more parameters as needed
                        }},
                        '[selector_name]'
                    )
                ]
            """
            ) 
            print("output-response", output['response'])
            # Split the response text into words (or use another tokenizer if needed)
            # Assuming output['response'] is a string, we split it into words
            # Removing all non-alphanumeric characters except underscores and spaces
            cleaned_response = re.sub(r"[^a-zA-Z0-9_\s]", "", output['response'].lower())

            # Split the cleaned response into words based on spaces
            response_words = set(cleaned_response.split())  # Split by whitespace after cleaning

            # Print cleaned response and response words for debugging
            print("Cleaned response:", cleaned_response)
            print("Response words:", response_words)

            # Now, find the matches
            matches = metaheuristics_search_terms.intersection(response_words)

            print("Matches found:", matches)


        while self.file_result != 0: # It needs to have a correct output

            # Second Prompt - - - - - - - - - - - - - - - - - - -
            second_output = ollama.generate(
            model = self.model,
            prompt=f"""
            {self.prompt} remember that: in the 'fun' variable you must change it too: 'fun = bf.{self.benchmark_function}({self.dimensions})', do not change these values given. 
            Add the following metaheuristic {output['response']} on the correct part of the template. DO NOT INVENT NEW WORDS OR TERMS.
            This is the following template (PLEASE FOLLOW STRICTYLY THE FOLLOWING TEMPLATE, DO NOT INVENT NEW CODE, DO NOT INVENT PARAMETERS, NOR OPERATORS, NOR SELECTORS NEITHER): 
            {data_template}
            If you encounter an error, address it as follows: {self.file_result_error}.
            """
            ) 


            output_response = self.execute_generated_code(second_output['response'], output_folder, number_iteration, False)
            #output_response = output['response']  ## N.S
        # Main Prompt - - - - - - - - - - - - - - - - - - -

        self.file_result = 1 # Necessary for new metaheuristics

        # Calling optuna - - - - - - - - - - - - - - - - - - -
        self.perform_optuna_tuning(output_folder, number_iteration)

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
        print("i am refinement")
            
        # Query for the Operators - - - - - - - - - - - - - - - - - - -
        output = ollama.embeddings(
        prompt=f"give me the best operators, operators' parameters and selectors for 'fun = bf.{self.benchmark_function}({self.dimensions})'",
        model=self.model_embed
        )
        results = self.python_collection.query(
        query_embeddings=[output["embedding"]],
        n_results=3
        )
        data = results['documents'][0][0]
        # Query for the Operators - - - - - - - - - - - - - - - - - - -
        
        # Query for the Template - - - - - - - - - - - - - - - - - - -
        output_template = ollama.embeddings(
        prompt="give me the given template to create a metaheuristic properly",
        model=self.model_embed
        )
        results = self.python_collection.query(
        query_embeddings=[output_template["embedding"]],
        n_results=2
        )
        data_template = results['documents'][0][0]
        #print("data-from-py-collection----", data)
        # Query for the Template - - - - - - - - - - - - - - - - - - -
        
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
        while self.file_result != 0:
            # Debugging data and feedback
            # print(f"Loop Iteration {checker_variable}, Data: {data}, Feedback: {relevant_feedback}")

            output = ollama.generate(
                model=self.model,
                prompt = f"""
                {self.prompt}, taking this template: {metaheuristic_code_file}, modify it in order to put these parameters.
                Use the following parameters for the search operators:
                Parameters: {self.hyperparameters}, do NOT modify anything else. 

                If you encounter an error, address it as follows: {self.file_result_error}.
                You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response. All outputs must be plain text. 
                """
            )
            output_response_refinement = self.execute_generated_code(output['response'], output_folder, number_iteration, False)
        # Refining the Metaheuristic - - - - - - - - - - - - - - - - - - 

        self.file_result = 1 # wrong again in order to work correctly 

        #temperature=0.7
 
        current_directory = os.getcwd()
        relative_path = "outputs-results"
        base_path_m = os.path.join(current_directory, relative_path)
        folder_name_m = self.folder_name
        file_name_m = f"execution_iteration_{number_iteration}.py"

        refined_meta_file_path = os.path.join(base_path_m, folder_name_m, file_name_m)

        with open(refined_meta_file_path, 'r', encoding='utf-8') as file:
            refined_metaheuristic_code_file = file.read()
        
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
                print(hyperparameters_str)  # Optional: Print the result

                # is there a better way to store the feedback? , how can I take a look on it??
                self.feedback_collection.add(
                    documents=[refined_metaheuristic_code_file, hyperparameters_str, str(self.performance_found)],
                    metadatas=[{'source': 'metaheuristic_code_file'}, {'source': 'parameters'}, {'source': 'performance_found'}],
                    ids=[f"{numero}_id_metaheuristic", f"{numero}_id_optuna_parameters", f"{numero}_id_performance_found"]
                )
            # Hyperparameters to feedback - - - - - - - - - - - - - - - - - - -


        """ 
        query_embedding = ollama.embeddings(model=self.model_embed, prompt="give me the metaheuristics with the best performance")
        # Set the number of feedback results to retrieve
        n_results = max(1, min(number_iteration, 7))
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        print("relevant,feedback", relevant_feedback)
        print("Prompt being passed:", f"Using this data: ...  Respond to this prompt: {self.prompt}, these are the hy: {self.hyperparameters}")
        # generate a response combining the prompt and data we retrieved in step 2
         """

        
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

        
        
        # Performance - - - - - - - - - - - - - - - - - - -
        if self.performance_found < self.best_performance:
            self.best_performance = self.performance_found
        # Performance - - - - - - - - - - - - - - - - - - -

 
        return output_response_refinement
    
    """
    perform_optuna_tuning: 
    - Must get the hyperparameters 
    """
    def perform_optuna_tuning(self, output_folder, number_iteration):
        print("i am perform_optuna_tuning")
        # Query for the Optuna Template - - - - - - - - - - - - - - - - - - -
        optuna_template = ollama.embeddings(
        prompt="give me the optuna template",
        model=self.model_embed
        )
        results = self.python_collection.query(
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

        # Query for the Optuna File Generation - - - - - - - - - - - - - - - - - - -
        optuna_task = f"""
        You are a highly skilled computer scientist specializing in natural computing. 
        Your task is to provide code strictly adhering to the specified template, without any deviations or additional content. 

        ### Rules and Requirements:
        1. **No Markdown or Triple Backticks**: Do not include any Markdown syntax or triple backticks (```) in your response. All outputs must be plain text.
        2. **Exact Code Compliance**: 
        - You must write exactly the provided code template.
        - Do not add, invent, or modify anything beyond the explicitly allowed sections.
        - Avoid adding any extra words, explanations, or paragraphs.

        ### Template and Modifications:
        - Follow this structure **precisely**, ensuring it matches the given template: 
        {optuna_template}

        ### Modifications Required:
        1. **Objective Function**: Replace the relevant part with:
            def objective(trial):
                heur = [
                    {extracted_metaheuristic} 
                ]
            Remeber that the format for the metaheuristic must be:
            heur = [
                    (  # Search operator 1
                        '[operator_name]',
                        {{
                            'parameter1': value1,
                            'parameter2': value2,
                            more parameters as needed
                        }},
                        '[selector_name]'
                    ),
                    (
                        '[operator_name]',
                        {{
                            'parameter1': value1,
                            'parameter2': value2,
                            ... more parameters as needed
                        }},
                        '[selector_name]'
                    )
                ]
         2. **'fun' Variable**: Change the 'fun' variable to: fun = bf.{self.benchmark_function}({self.dimensions})
        Your response must conform exactly to the provided instructions and template. Any deviation will be incorrect.
        """
        current_output_optuna = ollama.generate(
            model = self.model,
            prompt= f"""{optuna_task}
            """
        )
        execution_result_optuna = self.execute_generated_code(current_output_optuna['response'], output_folder, number_iteration, True)
        # Query for the Optuna File Generation - - - - - - - - - - - - - - - - - - -

        # While for the Optuna File Generation - - - - - - - - - - - - - - - - - - -
        while self.file_result != 0: 
            output = ollama.generate(
            model = self.model,
            prompt=f"""{optuna_task}
            If you encounter an error, fix it, these are the errors: {self.file_result_error}            
            """
            ) 
            if print(output['response']) != "":
                checker_variable += 1
            print("checker_variable--OPTUNA--->>>>>", checker_variable)

            execution_result_optuna = self.execute_generated_code(output['response'], output_folder, number_iteration, True)
        # While for the Optuna File Generation - - - - - - - - - - - - - - - - - - -

        self.file_result = 1 # Resetting the variable

        # Getting the Hyperparameters - - - - - - - - - - - - - - - - - - -
        current_directory = os.getcwd()
        folder_name = self.folder_name
        file_name = f"execution_optuna_result_{number_iteration}.txt"

        file_path = os.path.join(current_directory, folder_name, file_name)
        self.hyperparameters, self.performance_found = self.get_preferential_values(file_path)
        print("hyperparameters", self.hyperparameters)
        print("performance_found", self.performance_found)
        # Getting the Hyperparameters - - - - - - - - - - - - - - - - - - -
        
        #  Important  - - - - - - - - - - - - - - - - - - -
        #Checking whether it should create again: 
        if self.performance_found < self.best_performance:
            #self.exploration(output_folder, number_iteration)
            
            #else:
            self.refinement(output_folder, number_iteration)
        #  Important  - - - - - - - - - - - - - - - - - - -

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

        return execution_result_optuna


    def execute_generated_code(self, code, output_folder, number_iteration, is_optuna):
        prefix = "execution_optuna_" if is_optuna else "execution_"
        # os.path.join()`: This function is used to create a proper file path string 
        # that works across different operating systems.
        file_name =  os.path.join(output_folder, f'{prefix}iteration_{number_iteration}.py')
        with open(file_name, 'w') as f:
            f.write(code)
        try:
            result = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=39)
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
            output_folder = output_folder_parent / f'ollama_output_{self.benchmark_function}({self.dimensions})_{timestamp}'
            self.folder_name = output_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            for i in range(self.max_iterations):
                self.logger.debug(f"Starting refinement iteration {i}")
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
    generator = GerateMetaheuristic("Rastrigin", 3, 15)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
    