import ollama
import chromadb
import os
import datetime
import subprocess
import logging
import re
import pathlib



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

"""

class GerateMetaheuristic:
    def __init__(self, benchmark_function, dimensions, max_iterations, model="myllama3:latest"):
        self.model = model
        self.max_iterations = max_iterations

        ## METAHEURISTIC CREATION
        self.benchmark_function = benchmark_function
        self.dimensions = dimensions

        ## LOGGER
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        ## ERROR CHECKER
        self.file_result = ""
        self.file_result_error = ""
        ## PERFORMANCE
        self.f_best = 0
        self.first_f_best = 15

        ## FOR OPTUNA
        self.file_contents = ""

        ## COLLECTIONS
        client = chromadb.Client()
        self.python_collection = client.create_collection(name="python_collection")
        self.feedback_collection = client.create_collection(name="feedback_collection")
        #self.optuna_collection = client.create_collection(name="optuna_collection")

        self.prompt = f"""You are a highly skilled computer scientist in the field of natural computing. Your task is to design a metaheuristic algorithm, 
        you should only use the information provided in the collection. Remember that when writing the operator's names, they should be ALL in LOWER CASE AND WITH A '_' 
        instead of typing a space. 
        In the 'fun' variable you must change it too: 'fun = bf.{self.benchmark_function}({self.dimensions})'
        In case there was an error, please fix it. This is the error: {self.file_result_error}.
        The metaheuristic must get a smaller fitness soultion, the actual fitness is: {self.f_best}.
        """""


        python_files_directory = 'llm-metaheuristics/metaheuristic_builder'
        optuna_files_dir = 'llm-metaheuristics/optuna_builder'

        for d in os.listdir(python_files_directory):
            file_path = os.path.join(python_files_directory, d)
            if os.path.isfile(file_path):  # Check if it's a file
                file_content = self.read_file(file_path)
                response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content)
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

        for d in os.listdir(optuna_files_dir):
            file_path = os.path.join(python_files_directory, d)
            if os.path.isfile(file_path):  # Check if it's a file
                file_content = self.read_file(file_path)
                response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content)
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

    def extract_code_from_code_with_optuna(self, code_file):
        response_optuna = ollama.embeddings(
        prompt="Take a look to the parameters of the operators and selectors provided.",
        model="mxbai-embed-large"
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
            prompt = f"""Modify this metaheuristic: {extracted_content}.So that if the parameter is a variable that provides a number, you must change it to:
            name_variable = trial.suggest_float('name_variable', 0.01, 0.9), 
            The O.01 and 0.9 provided is just an example. You can have different ranges, but please do not use ranges that:
            - Are above 0.9 if the variable is radius
            - Are above 25 if the variable is angle
            - Are above 3 if the variable is swarm_conf or self_conf
            ALWAYS use a coma after modifying or writing a variable parameter.
            PLEASE do not modify anything else of the, just do what I told you to. 
            Do not add any more words or paragraphs, just follow this instructions. 
            """
            ) 

            # Deleted from the prompt: , take a look to the other parameters provided for the operators and selectors: {optuna_data}

            print("is it doing it wrong", output['response'])
            return output['response']
        else:
            return None

        
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
            

    def self_refine(self, output_folder, number_iteration):
        current_output = ""
        checker_variable = 1 # variable to count how many times it has created a metaheuristic. 
        response = ollama.embeddings(
        prompt=self.prompt,
        model="mxbai-embed-large"
        )
        results = self.python_collection.query(
        query_embeddings=[response["embedding"]],
        n_results=1
        )
        data = results['documents'][0][0]

        # generate a response combining the prompt and data we retrieved in step 2
        output = ollama.generate(
        model = self.model,
        prompt = f"Using this data: {data}. Respond to this prompt: {self.prompt}"
        ) 
        self.execute_generated_code(output['response'], output_folder, number_iteration, False)
        #print("execution_result-need-to-see", execution_result)

        ## CHECKING THE FITNESS
        file_name =  os.path.join(output_folder, f'execution_result_{number_iteration}.txt')
        self.extract_best_performance(file_name)

        print( "num-iter----",number_iteration , "------", "self.f_best", self.f_best,  "-", self.first_f_best)
        # Must create a functionable and better-fitness metaheuristic

        #or (self.f_best > self.first_f_best)
        while self.file_result != 0 or self.f_best > self.first_f_best: 
            # generate a response combining the prompt and data we retrieved in step 2
            output = ollama.generate(
            model = self.model,
            prompt = f"Using this data: {data}. Respond to this prompt: {self.prompt}"
            ) 

            if print(output['response']) != "":
                checker_variable += 1
            print("checker_variable----->>>>>", checker_variable)

            self.execute_generated_code(output['response'], output_folder, number_iteration, False)
            current_output = output
            print("current_output-need-to-se", current_output)
            if checker_variable >= 3:
                print("Reached maximum iterations, exiting loop.")
                break
        
        self.first_f_best = self.f_best
        
        print("current_output-need-to-see-outside-while", current_output)
        
        ## FEEDBACK PROCESS: Must be after the while, since I must only store the metaheuristics that ran well. 
        feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=output['response'])
        self.feedback_collection.add(
            ids=[f"iteration_{number_iteration}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[output['response'] + "\n" ],
            metadatas={"f_best": self.f_best}
        )
        
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=output['response'])
        n_results = max(1, min(number_iteration, 7))
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        print("relevant_feedback", relevant_feedback)
        print("relevant_feedback['documents']}", relevant_feedback['documents'])

        return current_output


    def self_refine_with_optuna(self, output_folder, number_iteration):
        # Here the feedback is not provided, but saved for the next metaheuristics creations. 

        current_output = ""
        checker_variable = 1 # variable to count how many times it has created a metaheuristic. 
        input_file_path = os.path.join(output_folder, f'execution_iteration_{number_iteration}.py')
        with open(input_file_path, 'r') as f:
            self.file_contents = f.read()
        extracted_metaheuristic = self.extract_code_from_code_with_optuna(self.file_contents)


        optuna_task =  f"""You are a highly skilled computer scientist in the field of natural computing. Your task is to make the optuna algorithm of the given metaheuristic. 
        When writing the response You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response, 
        all outputs must be plain text. 
        You must write exactly the following code, DO NOT INVENT NEW THINGS, NOR WORDS, NOR PARAGRAPHS, NOR ANYTHING, FOLLOW THE EXACT TEMPLATE:

import optuna
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[3]
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
            prompt= f"""Follow this prompt: {optuna_task}, take into account that this was the given error {self.file_result_error},
            you should fix it. Do not add any more words or paragraphs, just follow these instructions. 
            """
        )

        execution_result_optuna = self.execute_generated_code(current_output_optuna['response'], output_folder, number_iteration, True)
        # execution_result_optuna will be the error, or trials. Must save it on the feeback if the output ran well (0)
        print("FIRST----execution_result_optuna--NEEDTOSE", execution_result_optuna)

        #or (self.f_best > self.first_f_best)
        while self.file_result != 0: 
            # generate a response combining the prompt and data we retrieved in step 2
            output = ollama.generate(
            model = self.model,
            prompt=f"Respond to this prompt: {optuna_task}"
            ) 

            if print(output['response']) != "":
                checker_variable += 1
            print("checker_variable--OPTUNA--->>>>>", checker_variable)

            execution_result_optuna = self.execute_generated_code(output['response'], output_folder, number_iteration, True)
            print("execution_result_optuna--NEEDTOSE", execution_result_optuna)
            current_output_optuna = output
            print("current_output_optuna-need-to-se", current_output_optuna)
            if checker_variable >= 3:
                print("Reached maximum iterations, exiting loop.")
                
        
        
        print("current_output_optuna-need-to-see-outside-while", current_output_optuna)
        
        ## FEEDBACK PROCESS: Must be after the while, since I must only store the metaheuristics that ran well. 
        feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output_optuna['response'])
        self.feedback_collection.add(
            ids=[f"iteration_{number_iteration}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[current_output_optuna['response'] + "\n" ]
        )
        
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output_optuna['response'])
        n_results = max(1, min(number_iteration, 7))
        relevant_feedback = self.feedback_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        print("relevant_feedback", relevant_feedback)
        print("relevant_feedback['documents']}", relevant_feedback['documents'])

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
        self.logger.debug("Starting main execution")
        try:
            # Create output folder
            self.logger.debug("Creating output folder")
            current_dir = pathlib.Path(__file__).parent.resolve()
            output_folder_parent = current_dir / 'outputs-results'

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = output_folder_parent / f'ollama_output_{self.benchmark_function}({self.dimensions})_{timestamp}'
            output_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new folder: {output_folder}")
            # Generate and refine the original output
            self.logger.debug("Starting________________")
        
            for i in range(self.max_iterations):
                self.logger.debug(f"Starting refinement iteration {i}")
                self.self_refine(output_folder, i)
                self.logger.info(f"Refined output for iteration {i} generated")

                self.self_refine_with_optuna(output_folder, i)

            # -------------------- 
            self.client.delete_collection(name="metaheuristic_builder")
            self.client.delete_collection(name="optuna_collection")
            self.client.delete_collection(name="feedback_collection")
        except Exception as e:
            self.logger.error(f"An error occurred in the main execution: {str(e)}")
            self.logger.exception("Exception details:")
            raise         

if __name__ == "__main__":
    generator = GerateMetaheuristic("Ackley1", 2, 7)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
    