import ollama
import chromadb
import os
import datetime
import subprocess
import logging
import re


"""
More straightforward information
Creates Optuna and metaheuristic,
model="deepseek-coder-v2" and "codegemma"
Added  prompt= f"Using this data {data} implement correctly the optuna library"
Instead of an "optuna prompt"

- Also modifies the output 

OPTUNA keeps adding .. ```python

"""
 

class NoCodeException(Exception):
    pass
 

class MetaheuristicGenerator:
    def __init__(self, benchmark_function, dimensions, model="deepseek-coder-v2", max_iterations=7):
        self.model = model
        self.max_iterations = max_iterations
        self.client = chromadb.Client()
        self.python_files_collection = self.client.create_collection(name="metaheuristic_builder")
        self.optuna_collection = self.client.create_collection(name="optuna_collection")
        self.feedback_collection = self.client.create_collection(name="feedback_collection")
        self.benchmark_function = benchmark_function
        self.dimensions = dimensions
        self.file_result = "" # Checking if output was succesful
        self.extracted_code = ""
        
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms."
        self.task_prompt =  f"""
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the bf.{self.benchmark_function}({self.dimensions}) optimization problem using only the operators and selectors from the parameters_to_take.txt file.

        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
        DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.

        INSTRUCTIONS:
        1. Use only the function: bf.{self.benchmark_function}({self.dimensions})
        2. Use only operators and selectors from parameters_to_take.txt.
        3. Use only the parameters of the operator chosen from parameters_to_take.txt.
        4. The options inside the array are the ones you can choose from to fill each parameter.
        5. Only use one variable per parameter
        6. Do Not use the whole array when writing the variable of the parameter.
        7. Write the variables without an array format
        8. Write the variable as a float or string format.
        9. The search space is between -1.0 (lower bound) and 1.0 (upper bound)
        10. Set num_iterations to 100
        12. Each operator must have its own selector
        13. Fill all parameters for the chosen operator with your best recommendations. You must read the complete parameters_to_take.txt file to know all the parameters for each operator.
        14. You can use Two operator per metaheuristic if you think that is the best option, but do not use more than three operators.
        15. Create only one metaheuristic per response
        16. DO NOT use any information or knowledge outside of what is provided in the parameters_to_take.txt file

        FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        # Name: [Your chosen name for the metaheuristic]
        # Code:
        import sys
        from pathlib import Path

        project_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_dir))
        import benchmark_func as bf
        import metaheuristic as mh

        fun = bf.{self.benchmark_function}({self.dimensions})   
        prob = fun.get_formatted_problem()

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

        met = mh.Metaheuristic(prob, heur, num_iterations=100)
        met.verbose = True
        met.run()

        print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

        # Short explanation and justification:
        # [Your explanation here, each line starting with '#']
        """ 

        self.optuna_refinement_prompt =  f"""
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the bf.{self.benchmark_function}({self.dimensions}) optimization problem using only the operators and selectors from the parameters_to_take.txt file.

        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
        DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.

        INSTRUCTIONS:
        1. Use only the function: bf.{self.benchmark_function}({self.dimensions})
        2. Use only operators and selectors from parameters_to_take.txt.
        3. Use only the parameters of the operator chosen from parameters_to_take.txt.
        4. The options inside the array are the ones you can choose from to fill each parameter.
        5. Only use one variable per parameter
        6. Do Not use the whole array when writing the variable of the parameter.
        7. Write the variables without an array format
        8. Write the variable as a float or string format.
        9. The search space is between -1.0 (lower bound) and 1.0 (upper bound)
        10. Set num_iterations to 100
        12. Each operator must have its own selector
        13. Fill all parameters for the chosen operator with your best recommendations. You must read the complete parameters_to_take.txt file to know all the parameters for each operator.
        14. You can use Two operator per metaheuristic if you think that is the best option, but do not use more than three operators.
        15. Create only one metaheuristic per response
        16. DO NOT use any information or knowledge outside of what is provided in the parameters_to_take.txt file

        FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

        # Name: [Your chosen name for the optuna-enhanced metaheuristic]
        # Code:

        import optuna
        import sys
        from pathlib import Path

        project_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_dir))

        import benchmark_func as bf
        import matplotlib.pyplot as plt

        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)
        import  population as pp
        import metaheuristic as mh
        import numpy as np
        from joblib import Parallel, delayed
        import multiprocessing

        # WRITE THE WHOLE FUNCTION
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

            # Note: If a word is in the code do not remove it, but if a number is in the code, replace it with "trial.suggest_float('variable_name', 0.1, 0.9)"
            def objective(trial):
                heur = [
                    using this code {self.extracted_code}

                ]

                fun = bf.{self.benchmark_function}({self.dimensions}) # This is the selected problem, the problem may vary depending on the case.
                prob = fun.get_formatted_problem()
                performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
                
                return performance

                # WRITE THE WHOLE CODE
                study = optuna.create_study(direction="minimize")  
                study.optimize(objective, n_trials=50) 

                print("Mejores hiperparámetros encontrados:")
                print(study.best_params)

                print("Mejor rendimiento encontrado:")
                print(study.best_value)   
        #  IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
        """
        self.python_files_dir = 'llm-metaheuristics/metaheuristic_builder'
        self.process_files(self.python_files_collection, self.python_files_dir)
        self.optuna_files_dir = 'llm-metaheuristics/optuna_builder'
        self.process_files(self.optuna_collection, self.optuna_files_dir)
    
    def process_files(self, name_collection, directory):
        for filename in os.listdir (directory):
            if filename.endswith('.py') or filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                file_content = self.read_file(file_path)

                response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content)
                embedding = response.get("embedding")
                
                if embedding:
                    name_collection.add(
                        ids=[filename],
                        embeddings=[embedding],
                        documents=[file_content],
                        metadatas=[{"filename": filename}]
                    )
                    print(f"Added {filename} to the optuna collection")
                else:
                    print(f"Warning: Empty embedding generated for {filename}")

        
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
        #self.client = LLMmanager(model)
        #self.model = model
        #self.f = f  # evaluation function, provides a string as feedback, a numerical value (higher is better), and a possible error string.
    
 
    
    def gen_optuna(self):
        self.task_prompt_optuna = f"""
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:
        Enhance the following metaheuristic code by creating a python file that incorporates Optuna for hyperparameter tuning:
        REMEMBER: 
        1. EVERY EXPLANATION MUST START WITH '#'. 
        2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```
        3. ONLY USE INFORMATION FROM THE optuna_builder folder (the one in the optuna_collection) and the information provided in this prompt.
        4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
        5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
        6. If you ever use genetic crossover, you must use genetic mutation as well. 
        8. Checking for any logical errors or inconsistencies.
        9. Improving the explanation and justification.   
        """

        full_prompt_optuna = self.task_prompt_optuna  # too possible to combine things 
        return full_prompt_optuna
    
    def extract_code_from_code(self, code_file):
        pattern = r'heur\s*=\s*\[(.*?)\]'  # Match content inside heur = [ ]
        match = re.search(pattern, code_file, re.DOTALL)

        if match:
            extracted_content = match.group(1).strip()  # Extract the code block
            return extracted_content
        else:
            return None
            
    def self_refine(self, initial_prompt, data, output_folder, number_iteration):
        current_output = ollama.generate(
            model="codegemma",
            prompt=f"Using this data: {data}. Respond to this prompt: {initial_prompt}"
        )

        print("printeando la respuesta, avr si hay error")
        print(current_output['response'])
        execution_result = self.execute_generated_code(current_output['response'], output_folder, number_iteration, False)
        
        feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'] + execution_result)
        self.feedback_collection.add(
            ids=[f"iteration_{number_iteration}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[current_output['response'] + "\n" + execution_result],
            metadatas=[{"iteration": number_iteration}]
        )
        
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'])
        
        n_results = max(1, min(number_iteration, 7))
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        
        """ 
        #self.python_files_collection.count() > 0:
        total_docs = self.python_files_collection.count()
        relevant_files = self.python_files_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=total_docs  # Retrieve all documents
        )
            
        if 'distances' in relevant_files:
            sorted_indices = sorted(range(len(relevant_files['distances'][0])), 
                                    key=lambda k: relevant_files['distances'][0][k])
            
            sorted_documents = [relevant_files['documents'][0][i] for i in sorted_indices]
            relevant_files['documents'] = [sorted_documents]
        
        max_docs_to_include = 3  # Adjust this number as needed
        relevant_files['documents'][0] = relevant_files['documents'][0][:max_docs_to_include]
        relevant_files = {"documents": ["No relevant Python files found."]}
        """
        print("self.file_result_NORMAL", self.file_result)
        while self.file_result != 0: 
            generated_meataheuristic = self.extract_code_from_code(current_output['response'])
        
        
            # Construct the refinement prompt with relevant feedback and Python collection (metaheuristic)
            refinement_prompt = f"""
            IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
            DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
            You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:

            {current_output['response']}
            The code was executed with the following result:
            {execution_result}
            You must fix the results. I need the metaheuristic to run correctly. 
            Here is relevant feedback from previous iterations:
            {relevant_feedback['documents']}

            Please analyze this output and suggest improvements and corrections. 
            Please DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ``` in the response.
            Please DO NOT USE ANY operators or parameters that are not in the parameters_to_take.txt file.
            This is the parameters_to_take.txt file:
            {data}
            IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
            Use the same template as the one provided before, which is:
            IMPORTANT:  Please create a metaheuristic different than {generated_meataheuristic}
            

            # Name: [Your chosen name for the metaheuristic]
            # Code:

            import sys
            from pathlib import Path

            project_dir = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(project_dir))

            import benchmark_func as bf
            import metaheuristic as mh


            fun = bf.{self.benchmark_function}({self.dimensions})
            prob = fun.get_formatted_problem()

            heur = [
                ( # Search operator 1
                '[operator_name]',
                {{ 
                    'parameter1': value1,
                    'parameter2': value2,
                    ... more parameters as needed
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

            met = mh.Metaheuristic(prob, heur, num_iterations=100)
            met.verbose = True
            met.run()
            print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))


            # Short explanation and justification:
            # [Your explanation here, each line starting with '#']

            REMEMBER: 
            1. EVERY EXPLANATION MUST START WITH '#'. 
            2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```.
            3. ONLY USE INFORMATION FROM THE parameters_to_take.txt FILE.
            DO NOT INVENT ANY NEW INFORMATION.
            4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
            5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
            6. If you ever use genetic crossover, you must use genetic mutation as well. 
            7. Verifying that only operators and parameters from parameters_to_take.txt are used.
            8. Checking for any logical errors or inconsistencies.
            9. Improving the explanation and justification.

            Provide your refined version of the entire output, not just the changes.
            """

            refined_output = ollama.generate(
            model="deepseek-coder-v2",
            prompt=refinement_prompt
            )
            
            # Code repeats itself
            self.execute_generated_code(refined_output['response'], output_folder, number_iteration, False)
            
            
            #current_output = refined_output
            
            return refined_output['response']

    
    
    def self_refine_with_optuna(self, model, output_folder, iteration_number):
        input_file_path = os.path.join(output_folder, f'execution_iteration_{iteration_number}.py')
        with open(input_file_path, 'r') as f:
            file_contents = f.read()
        self.extracted_code = self.extract_code_from_code(input_file_path)
        print("letsee --- input_file_path", input_file_path)
        current_output_optuna = ollama.generate(
            model = model,
            prompt=f"Using this information of the file generated before: {file_contents}. Respond to this prompt: {self.role_prompt} {self.optuna_refinement_prompt}"
        )

        print("printeando la respuesta, optuna")
        print(current_output_optuna['response'])
        self.execute_generated_code(current_output_optuna['response'], output_folder, iteration_number, True)

        print("self.file_result_optuna", self.file_result)
        while self.file_result != 0: 
            response = ollama.embeddings(
                            prompt = f"{self.role_prompt} {self.optuna_refinement_prompt}",
                            model="mxbai-embed-large"
                        )
            results = self.optuna_collection.query(
                    query_embeddings=[response["embedding"]],
                    n_results=1
                )
            optuna_data = results['documents'][0][0]
        
            current_output_optuna = ollama.generate(
                model=model,
                prompt=f"Using this file: {input_file_path}, and this data: {optuna_data} Respond to this prompt: {self.role_prompt} {self.optuna_refinement_prompt}"
            )        

            self.execute_generated_code(current_output_optuna['response'], output_folder, iteration_number, True) 
            return current_output_optuna['response']
        
        

    def execute_generated_code(self, code, output_folder, iteration, is_optuna):
        prefix = "execution_optuna_" if is_optuna else "execution_"
        # os.path.join()`: This function is used to create a proper file path string 
        # that works across different operating systems.
        file_name =  os.path.join(output_folder, f'{prefix}iteration_{iteration}.py')
        #print(file_path)
        with open(file_name, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=100)
            execution_result = f"Exit code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            self.file_result = result.returncode
            result_file_name = f'{prefix}result_{iteration}.txt'
            result_file_path = os.path.join(output_folder, result_file_name)
            with open(result_file_path, 'w') as f:
                f.write(execution_result)
            return execution_result
        except subprocess.TimeoutExpired:
            return "Execution timed out after 30 seconds"
        except Exception as e:
            return f"An error occurred during execution: {str(e)}"
            

    # do we need iterations here?
    def construct_refinement_prompt(self, current_output, execution_result, data, feedback_embedding):
        self.task_prompt = f"""
        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
        DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:

        {current_output['response']}
        The code was executed with the following result:
        {execution_result}
        You must fix the results. I need the metaheuristic to run correctly. 
        Here is relevant feedback from previous iterations:
        {feedback_embedding['documents']}

        Here are relevant Python files that might be helpful:
        {data['documents']}

        Please analyze this output and suggest improvements and corrections. 
        Please DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ``` in the response.
        Please DO NOT USE ANY operators or parameters that are not in the parameters_to_take.txt file.
        This is the parameters_to_take.txt file:
        {data}
        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
        Use the same template as the one provided before, which is:
        
        # Name: [Your chosen name for the metaheuristic]
        # Code:
        import sys
        from pathlib import Path

        project_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_dir))

        import benchmark_func as bf
        import metaheuristic as mh


        fun = bf.Rastrigin(2)
        prob = fun.get_formatted_problem()

        heur = [
            ( # Search operator 1
            '[operator_name]',
            {{ 
                'parameter1': value1,
                'parameter2': value2,
                 ... more parameters as needed
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

        met = mh.Metaheuristic(prob, heur, num_iterations=100)
        met.verbose = True
        met.run()
        print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))


        # Short explanation and justification:
        # [Your explanation here, each line starting with '#']

        REMEMBER: 
        1. EVERY EXPLANATION MUST START WITH '#'. 
        2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```.
        3. ONLY USE INFORMATION FROM THE parameters_to_take.txt FILE.
        DO NOT INVENT ANY NEW INFORMATION.
        4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
        5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
        6. If you ever use genetic crossover, you must use genetic mutation as well. 
        7. Verifying that only operators and parameters from parameters_to_take.txt are used.
        8. Checking for any logical errors or inconsistencies.
        9. Improving the explanation and justification.

        Provide your refined version of the entire output, not just the changes.
        """
        full_prompt = f"{self.task_prompt}"
        return full_prompt

    def run(self):
        self.logger.debug("Starting main execution")
        try:
            prompt = self.role_prompt + self.task_prompt
            
            response = ollama.embeddings(
                prompt=prompt,
                model="mxbai-embed-large"
            )
            # Check if the embedding is empty
            if not response.get("embedding"):
                self.logger.error("Generated embedding is empty")
        
            
            results = self.python_files_collection.query(
                query_embeddings=[response["embedding"]],
                n_results=1
            )
            data = results['documents'][0][0]


            """
            optuna_prompt = self.optuna_refinement_prompt()

            optuna_response = ollama.embeddings(
                prompt=optuna_prompt,
                model="mxbai-embed-large"
            )
            self.logger.debug("Querying Optuna collection")
            optuna_results = self.optuna_collection.query(
                query_embeddings=[optuna_response["embedding"]],
                n_results=1
            )
            optuna_data = optuna_results['documents'][0][0]
            """  

            # Create output folder
            self.logger.debug("Creating output folder")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(current_dir, f'ollama_output_{timestamp}')

            os.makedirs(output_folder)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            max_iterations = 3
            for i in range(max_iterations):
                self.logger.debug(f"Starting refinement iteration {i}")
                self.self_refine(prompt, data, output_folder, i)
                self.logger.info(f"Refined output for iteration {i} generated")

                self.self_refine_with_optuna("codegemma", output_folder, i)
                #self.self_refine_with_optuna(optuna_prompt, "codegemma", output_folder, i)
            self.logger.debug("Main execution completed")
            self.client.delete_collection(name="metaheuristic_builder")
            self.client.delete_collection(name="optuna_collection")
            self.client.delete_collection(name="feedback_collection")
        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise



            

if __name__ == "__main__":
    generator = MetaheuristicGenerator("Rastrigin", 2)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
    

