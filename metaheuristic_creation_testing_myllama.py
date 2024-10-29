import sys
import ollama
import chromadb
import os
import datetime
import subprocess
import logging
import re
from ollama import ResponseError


class NoCodeException(Exception):
    pass

class MetaheuristicGenerator:
    def __init__(self,number_experiment, dimensions, model="my-new-llama3", max_iterations=7):
        self.model = model
        self.max_iterations = max_iterations
        self.client = chromadb.Client()
        self.python_files_collection = self.client.create_collection(name="metaheuristic_builder", metadata={"hnsw:space": "cosine"})
        self.optuna_collection = self.client.create_collection(name="optuna_collection", metadata={"hnsw:space": "cosine"})
        self.feedback_collection = self.client.create_collection(name="feedback_collection", metadata={"hnsw:space": "cosine"})
        self.number_experiment = number_experiment
        self.dimensions = dimensions
        self.optuna_execution_result = ""
        
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        
        
        self.python_files_dir = 'llm-metaheuristics/metaheuristic_builder'
        self.process_python_files()
        self.optuna_files_dir = 'llm-metaheuristics/optuna_builder'
        self.process_optuna_files()
    
    def process_optuna_files(self):
        for filename in os.listdir(self.optuna_files_dir):
            if filename.endswith('.py') or filename.endswith('.txt'):
                file_path = os.path.join(self.optuna_files_dir, filename)
                file_content = self.read_file(file_path)

                response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content)
                embedding = response.get("embedding")
                
                if embedding:
                    self.optuna_collection.add(
                        ids=[filename],
                        embeddings=[embedding],
                        documents=[file_content],
                        metadatas=[{"filename": filename}]
                    )
                    print(f"Added {filename} to the optuna collection")
                else:
                    print(f"Warning: Empty embedding generated for {filename}")


    def process_python_files(self):
        for filename in os.listdir(self.python_files_dir):
            if filename.endswith('.py') or filename.endswith('.txt'):
                file_path = os.path.join(self.python_files_dir, filename)
                file_content = self.read_file(file_path)
                
                response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content)
                embedding = response.get("embedding")
                
                if embedding:
                    self.python_files_collection.add(
                        ids=[filename],
                        embeddings=[embedding],
                        documents=[file_content],
                        metadatas=[{"filename": filename}]
                    )
                    print(f"Added {filename} to the python collection")
                else:
                    print(f"Warning: Empty embedding generated for {filename}")
        
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
    
    def generate_prompt(self):
        self.task_prompt = f"""
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the optimization problem using only the operators and selectors from the parameters_to_take.txt file.

        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
        DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.

        INSTRUCTIONS:
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
        - Do not write anything before the template - 
        FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        # Name: [Your chosen name for the metaheuristic]
        # Code:
        import sys
        sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
        import numpy as np
        import metaheuristic as mh
        import ioh
        from P1 import P1
    
def evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas):
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    met = mh.Metaheuristic(prob, heur, num_agents=num_agents, num_iterations=num_iterations)
    met.verbose = True
    met.run()
    best_position, f_best = met.get_solution()
    return f_best, best_position


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

                
        problem_id= {self.number_experiment}
        instance = 1
        dimension = {self.dimensions}
        num_agents = 100
        num_iterations = 400
        num_replicas = 1

        evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

        problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
        optimal_fitness = problem.optimum.y


        # Short explanation and justification:
        # [Your explanation here, each line starting with '#']
        """

        full_prompt = self.task_prompt  
        return full_prompt
    
    def generate_optuna_prompt(self):
        self.task_prompt_optuna = f"""
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:
        Enhance the following metaheuristic code by creating a python file that incorporates Optuna for hyperparameter tuning:
        REMEMBER: 
         - Do not write anything before the template - 
        1. EVERY EXPLANATION MUST START WITH '#'. 
        2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```
        3. ONLY USE INFORMATION FROM THE optuna_builder folder (the one in the optuna_collection) and the information provided in this prompt.
        4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
        5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
        6. If you ever use genetic crossover, you must use genetic mutation as well. 
        8. Checking for any logical errors or inconsistencies.
        9. Improving the explanation and justification.   
        """

        full_prompt_optuna = self.task_prompt_optuna   
        return full_prompt_optuna
            
    def self_refine(self, initial_prompt, data, output_folder, number_iteration):
        """
        Repetition in LLMs is mostly caused by the greedy approach of choosing the next token,
        where the highest probability token is always selected. These repetitions can be 
        prevented by introducing a penalty that discounts the scores of tokens that have 
        been generated before. 
        """
    
        current_output = ollama.generate(
            model="my-new-llama3",
            prompt=f"Using this data: {data}. Respond to this prompt: {initial_prompt}"
        )
        # Write initial output  # commenting to avoid to much files 
        #write_output_to_file(current_output['response'], output_folder, 0)
        execution_result = self.execute_generated_code(current_output['response'], output_folder, number_iteration, False)
        
        # Add the current output and execution result to the feedback collection
        feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'] + execution_result)
        self.feedback_collection.add(
            ids=[f"iteration_{number_iteration}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[current_output['response'] + "\n" + execution_result],
            metadatas=[{"hnsw:space": "cosine"}]
        )
        
        # Retrieve relevant feedback from previous iterations
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'])
        
        # Ensure n_results is at least 1
        n_results = max(1, min(number_iteration, 7))
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        
       
        # Construct the refinement prompt with relevant feedback and Python files
        refinement_prompt = f"""
        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
        DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
        You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:

        {current_output['response']}
        The code was executed with the following result:
        {execution_result} and the following optuna result which shows the performance and the best parameters found, take into account this information for the next enhancement iteration {self.optuna_execution_result}
        You must fix the results. I need the metaheuristic to run correctly. 
        Here is relevant feedback from previous iterations:
        {relevant_feedback['documents']}

         
        Please analyze this output and suggest improvements and corrections. 
        Please DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ``` in the response.
        Please DO NOT USE ANY operators or parameters that are not in the parameters_to_take.txt file.
        This is the parameters_to_take.txt file:
        {data}
        IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
        Use THIS SAME template to generate the next iteration, which is:
        - Do not write anything before the template - 
        # Name: [Your chosen name for the metaheuristic]
        # Code:

        import sys
        sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
        import numpy as np
        import metaheuristic as mh
        import ioh
        from P1 import P1


        def evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas):
            ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
            fun = P1(variable_num=dimension, problem=ioh_problem)
            prob = fun.get_formatted_problem()
            met = mh.Metaheuristic(prob, heur, num_agents=num_agents, num_iterations=num_iterations)
            met.verbose = True
            met.run()
            best_position, f_best = met.get_solution()
            return f_best, best_position

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

        problem_id= {self.number_experiment}
        instance = 1
        dimension = {self.dimensions}
        num_agents = 100
        num_iterations = 400
        num_replicas = 1

        evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)

        problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
        optimal_fitness = problem.optimum.y


        # Short explanation and justification:
        # [Your explanation here, each line starting with '#']

        REMEMBER: 
        PLEASE DO NOT WRITE ANYTHING BEFORE THIS NEXT GIVEN TEMPLATE:
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
        """

        refined_output = ollama.generate(
        model="my-new-llama3",
        prompt=refinement_prompt
        )
        
        # Write refined output
        self.execute_generated_code(refined_output['response'], output_folder, number_iteration, False)
        
       
        #current_output = refined_output
     
        return refined_output['response']
    
    def extract_code_from_code(self, code_file):
        with open(code_file, 'r') as file:
            content = file.read()
        # Use regex to extract content between 'heur[' and ']'
        # Regex pattern to capture content inside 'heur = [' and ']'
        pattern = r'heur\s*=\s*\[(.*?)\]'  # Match content inside heur = [ ]
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted_content = match.group(1).strip()  # Extract the code block
            return extracted_content
    
 

    def self_refine_with_optuna(self, optuna_promt, model ,  output_folder, iteration_number):
        input_file_path = os.path.join(output_folder, f'execution_iteration_{iteration_number}.py')
        
        with open(input_file_path, 'r') as file:
            original_code = file.read()

            # Retrieve all Optuna files
        if self.optuna_collection.count() > 0:
            total_docs_optuna = self.optuna_collection.count()
            relevant_files_optuna = self.optuna_collection.query(
                query_embeddings=[ollama.embeddings(prompt=original_code, model="snowflake-arctic-embed")['embedding']],
                n_results=total_docs_optuna  # Retrieve all documents
            )
            
            # Sort the results by relevance score (if available)
            if 'distances' in relevant_files_optuna:
                sorted_indices = sorted(range(len(relevant_files_optuna['distances'][0])), 
                                        key=lambda k: relevant_files_optuna['distances'][0][k])
                
                sorted_documents = [relevant_files_optuna['documents'][0][i] for i in sorted_indices]
                relevant_files_optuna['documents'] = [sorted_documents]
            
            # Limit the number of documents to include in the prompt if necessary
            max_docs_to_include = 3  # Adjust this number as needed
            relevant_files_optuna['documents'][0] = relevant_files_optuna['documents'][0][:max_docs_to_include]
            print("veamos", relevant_files_optuna['documents'][0])
        else:
            relevant_files_optuna = {"documents": ["No relevant optuna files found."]}
        print("viendoooo;;;;", self.extract_code_from_code(input_file_path))

        self.full_prompt_with_optuna = f"""
Please add Optuna to optimize the parameters of the GIVEN METAHEURISTIC. 
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.

 - Do not write anything before the template - 
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
FOLLOW EXACTLY the following template for the optuna-enhanced metaheuristic:
PLEASE FOLLOW EXACTLY the following template for the optuna-enhanced metaheuristic:

DO NOT WRITE ANYTHING BEFORE THIS NEXT GIVEN TEMPLATE:
PLEASE DO NOT WRITE ANYTHING BEFORE THIS NEXT GIVEN TEMPLATE:
- Do not write anything before the template - 
THIS IS THE TEMPLATE:
# Name: [Your chosen name for the optuna-enhanced metaheuristic]
# Code:

import optuna
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import  population as pp
import metaheuristic as mh
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from P1 import P1

    
problem_id = {self.dimensions}
instance = 1
dimension = 5
num_agents = 100
num_iterations = 400
num_replicas = 1


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
        {self.extract_code_from_code(input_file_path)}
    ]

                
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()
    performance = evaluate_sequence_performance(heur, prob, num_agents=50, num_iterations=100, num_replicas=30)
    
    return performance

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 

print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print("Mejor rendimiento encontrado:")
print(study.best_value)

 - Do not write anything before the template - 
#  IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```. ALL OUTPUT MUST BE PLAIN TEXT.
        """
        refined_output = ollama.generate(
        model="my-new-llama3",
        prompt=self.full_prompt_with_optuna,
        )
        
        # Write refined output
        self.execute_generated_code(refined_output['response'], output_folder, iteration_number, True)
             
        return refined_output['response']
    
        

    def execute_generated_code(self, code, output_folder, iteration, is_optuna):
        prefix = "execution_optuna_" if is_optuna else "execution_"
        # os.path.join()`: This function is used to create a proper file path string 
        # that works across different operating systems.
        file_name =  os.path.join(output_folder, f'{prefix}iteration_{iteration}.py')
        #file_path = os.path.join(output_folder,  f'{prefix}{file_name}.py')
        #print(file_path)
        with open(file_name, 'w') as f:
            f.write(code)
        
        try:
            if is_optuna:
                optuna_output = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=70)
                self.optuna_execution_result = f"Exit code Optuna: {optuna_output.returncode}\nStdout:\n{optuna_output.stdout}\nStderr:\n{optuna_output.stderr}"
                result_file_name = f'{prefix}result_optuna{iteration}.txt'
                result_file_path = os.path.join(output_folder, result_file_name)
                with open(result_file_path, 'w') as f:
                    f.write(self.optuna_execution_result)
            else:

                result = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=60)
                execution_result = f"Exit code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
                
                result_file_name = f'{prefix}result_{iteration}.txt'
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
            prompt = self.generate_prompt()
            
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



            optuna_prompt = self.generate_optuna_prompt()

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

            # Create output folder
            self.logger.debug("Creating output folder")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(current_dir, f'ollama_output_{timestamp}')

            os.makedirs(output_folder)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            max_iterations = 7
            for i in range(max_iterations):
                self.logger.debug(f"Starting refinement iteration {i}")
                self.self_refine(prompt, data, output_folder, i)
                self.logger.info(f"Refined output for iteration {i} generated")

                #self.self_refine_with_optuna(optuna_prompt, "codegemma", output_folder, i)
                self.self_refine_with_optuna(optuna_prompt, "myllama3:latest", output_folder, i)
            self.logger.debug("Main execution completed")
            self.client.delete_collection(name="metaheuristic_builder")
            self.client.delete_collection(name="optuna_collection")
            self.client.delete_collection(name="feedback_collection")
        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise



            

if __name__ == "__main__":
    generator = MetaheuristicGenerator(4, 2)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
    


