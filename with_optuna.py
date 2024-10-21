# This code is used to create a metaheuristic for an optimization problem using optuna and ollama.
#import optuna
import ollama
import chromadb
import numpy as np
import os
import benchmark_func as bf
import sys
import datetime
import subprocess
#import time
import logging
from sklearn.neighbors import NearestNeighbors
#import nltk
#from mattsollamatools import chunker

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# sys is a built-in Python module that provides access to some variables and functions used or maintained by the Python interpreter.
# sys.path is used to add directories to the Python interpreter's search path for modules.
# append is a method that adds a directory to the end of the search path.
# 'llm-metaheuristics/algorithm_creation' is the directory to be added to the search path.
sys.path.append('llm-metaheuristics/algorithm_creation')

# Define the function
fun = bf.HappyCat(2)
# Get the function name
# fun_name = fun.__class__.__name__

# Initialize ChromaDB client: AI-native open-source vector database
client = chromadb.Client()


def read_python_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Directory containing Python files
python_files_dir = 'llm-metaheuristics/algorithm_creation'
optuna_files_dir = 'llm-metaheuristics/optuna_builder'

 # Create a collection for Python files
 # Collections are where you'll store your embeddings, documents, and any additional metadata. 

collection = client.create_collection(name="algorithm_creation") # Create a collection for Python files
optuna_collection = client.create_collection(name="optuna_builder") # Create a collection for Optuna files

# Process each Python file in the directory
for filename in os.listdir(python_files_dir):
    if filename.endswith('.py') or filename.endswith('.txt'):
        file_path = os.path.join(python_files_dir, filename)
        file_content = read_python_file(file_path)
        
        # Embeddings are numerical representations of your data that can be used for tasks like similarity search, 
        # clustering, and more.
        response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content)
        embedding = response.get("embedding")
        
        if embedding:
            collection.add(
                ids=[filename],
                embeddings=[embedding],
                documents=[file_content],
                metadatas=[{"filename": filename}]
            )
        else:
            print(f"Warning: Empty embedding generated for {filename}")

# Process each optuna file in the directory
for filename in os.listdir(optuna_files_dir):
    if filename.endswith('.py') or filename.endswith('.txt'):
        file_path_optuna = os.path.join(optuna_files_dir, filename)
        file_content_optuna = read_python_file(file_path_optuna)
        
        optuna_response = ollama.embeddings(model="mxbai-embed-large", prompt=file_content_optuna)
        optuna_embedding = optuna_response.get("embedding")
        
        if optuna_embedding:
            optuna_collection.add(
                ids=[filename],
                embeddings=[optuna_embedding],
                documents=[file_content_optuna],
                metadatas=[{"filename": filename}]
            )
        else:
            print(f"Warning: Empty embedding generated for {filename}")

prompt = """
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
        DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.

You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the {fun} optimization problem using only the operators and selectors from the parameters_to_take.txt file.

INSTRUCTIONS:
1. Use only the function: bf.HappyCat(2)
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

These are the parameters to take, depending on the selected operator, remember YOU MUST ONLY USE USE ONE VARIABLE PER PARAMETER, DO NOT USE THE WHOLE ARRAY, and write the variable without an array format, but as a float or string format:
{
  "random_search": {   # this operator only has these next parameters
    { # parameters
      "scale": 1.0 or 0.01,
      "distribution": "uniform" or "gaussian" or "levy"
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "central_force_dynamic": {   # this operator only has these next parameters
    { # parameters
      "gravity": 0.001,
      "alpha": 0.01,
      "beta": 1.5,
      "dt": 1.0
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "differential_mutation": {  # this operator only has these next parameters
    { # parameters
      "expression": "rand" or "best" or "current" or  "current-to-best" or "rand-to-best" or "rand-to-best-and-current",
      "num_rands": 1,
      "factor": 1.0
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "firefly_dynamic": {  # this operator only has these next parameters
    { # parameters
      "distribution": "uniform" or "gaussian" or "levy",
      "alpha": 1.0,
      "beta": 1.0,
      "gamma": 100.0
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "genetic_crossover": {  # this operator only has these next parameters
    { # parameters
      "pairing": "rank" or "cost" or "random" or"tournament_2_100",
      "crossover": "single" or "two" or "uniform" or "blend" or "linear_0.5_0.5",
      "mating_pool_factor": 0.4
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "genetic_mutation": {  # this operator only has these next parameters
    { # parameters
      "scale": 1.0,
      "elite_rate": 0.1,
      "mutation_rate": 0.25,
      "distribution": "uniform" or "gaussian" or "levy"
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "gravitational_search": {  # this operator only has these next parameters
    { # parameters
      "gravity": 1.0,
      "alpha": 0.02
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "random_flight": {  # this operator only has these next parameters
    { # parameters
      "scale": 1.0,
      "distribution": "levy" or "uniform" or"gaussian",
      "beta": 1.5
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "local_random_walk": { # this operator only has these next parameters
    { # parameters    
      "probability": 0.75,
      "scale": 1.0,
      "distribution": "uniform" or "gaussian" or "levy"
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "random_sample": {  # this operator only has these next parameters
    { # parameters }
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "spiral_dynamic": {  # this operator only has these next parameters
    { # parameters
      "radius": 0.9,
      "angle": 22.5,
      "sigma": 0.1
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  "swarm_dynamic": {  # this operator only has these next parameters
   { # parameters
      "factor": 0.7 or 1.0,
      "self_conf": 2.54,
      "swarm_conf": 2.56,
      "version": "inertial" or "constriction",
      "distribution": "uniform" or "gaussian" or "levy"
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  }
}
Now create the metaheuristic:
# Name: [Your chosen name for the metaheuristic]
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh


fun = bf.HappyCat(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1
            '[operator_name]',
            {
                'parameter1': value1,
                'parameter2': value2,
                 ... more parameters as needed
            },
            '[selector_name]'
            ),
            (   # Search operator 2
            '[operator_name]',
            {
                'parameter1': value1,
                'parameter2': value2,
                 ... more parameters as needed
            },
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
2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```
3. ONLY USE INFORMATION FROM THE parameters_to_take.txt FILE.
4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
6. If you ever use genetic crossover, you must use genetic mutation as well. 
7. Verifying that only operators and parameters from parameters_to_take.txt are used.
8. Checking for any logical errors or inconsistencies.
9. Improving the explanation and justification.

"""

optuna_prompt = """
You are a computer scientist specializing in natural computing and metaheuristic algorithms. 
You have been tasked with refining and improving the given metaheuristic code.
Enhance the following metaheuristic code by incorporating Optuna for hyperparameter tuning:
REMEMBER: 
1. EVERY EXPLANATION MUST START WITH '#'. 
2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```
3. ONLY USE INFORMATION FROM THE optuna_builder folder (the one in the optuna_collection) and the information provided in this prompt.
4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
6. If you ever use genetic crossover, you must use genetic mutation as well. 
8. Checking for any logical errors or inconsistencies.
9. Improving the explanation and justification.

Please add Optuna to optimize the parameters of the GIVEN METAHEURISTIC. 
Ensure the Optuna-enhanced version still follows the original structure and logic.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
USE THIS SAME CODE, do not create any other code: 


FOLLOW EXACTLY the following template for the optuna-enhanced metaheuristic:
# Name: [Your chosen name for the optuna-enhanced metaheuristic]
# Code:

import optuna
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

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
            ('random_search', {
                'scale': trial.suggest_float('scale', 0.01, 1.0),
                'distribution': 'selected_distribution', # Do not remove or changethis word, it is used to select the distribution.
            }, 'selected_selector'), # Do not remove or change this word given, it is used to select the population.
            ('central_force_dynamic', {
                'gravity': trial.suggest_float('gravity', 0.001, 0.1),
                'alpha': trial.suggest_float('alpha', 0.01, 0.1),
                'beta': trial.suggest_float('beta', 1.0, 2.0),
                'dt': trial.suggest_float('dt', 0.01, 0.1)
            }, 'selected_selector'), # Do not remove or changet this word given, it is used to select the population.
            ("differential_mutation": { 
                "expression": "rand" or "best" or "current" or  "current-to-best" or "rand-to-best" or "rand-to-best-and-current",
                "num_rands": 1,
                "factor": 1.0
                }, 'selected_selector'), # Do not remove or changet this word given, it is used to select the population.
            ('genetic_crossover', {
                'pairing': 'selected_pairing',   # Do not remove or change this word, it is used to select the pairing.
                'crossover': 'selected_crossover',   # Do not remove or change this word, it is used to select the crossover.
                'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)  
            }, 'all'), # Do not remove or change this word given, it is used to select the population.
            ('swarm_dynamic', {
                'factor': trial.suggest_float('factor', 0.4, 0.9),
                'self_conf': trial.suggest_float('self_conf', 1.5, 3.0),
                'swarm_conf': trial.suggest_float('swarm_conf', 1.5, 3.0),
                'version': 'selected_version',  # Do not remove or change this word, it is used to select the version.
                'distribution': 'selected_distribution'  # Do not remove or change this word, it is used to select the distribution.
            }, 'all'), # Do not remove or change this word given, it is used to select the population.
            ('differential_mutation', {
                'expression': 'selected_expression', # Do not remove or changethis word, it is used to select the expression.
                'num_rands': trial.suggest_int('num_rands', 1, 3),
                'factor': trial.suggest_float('factor', 0.1, 1.0)
            }, 'all'), # Do not remove or change this word given, it is used to select the population.
            ('genetic_crossover', {
                'pairing': 'selected_pairing',   # Do not remove or change this word, it is used to select the pairing. 
                'crossover': 'selected_crossover',   # Do not remove or change this word, it is used to select the crossover.
                'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)  
            }, 'all'), # Do not remove or change this word given, it is used to select the population.
        ]
        fun = bf.HappyCat(2) # This is the selected problem, the problem may vary depending on the case.
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
    
# Short explanation and justification:
# [Your explanation here, each line starting with '#'] 
"""
 
def self_refine(initial_prompt, data, model, output_folder, iteration):
    # Stores feedback from previous iterations.
    feedback_collection = chromadb.Client().get_or_create_collection(name="feedback_collection")
    
    # Stores Python files.
    python_files_collection = chromadb.Client().get_or_create_collection(name="algorithm_creation")
    
    # Uses Ollama to generate an initial response based on the given prompt and data.
    current_output = ollama.generate(
        model=model,
        prompt=f"Using this data: {data}. Respond to this prompt: {initial_prompt}"
    )
    
    #print(current_output['response'])
    
    # Write output
    #execute_generated_code(current_output['response'], output_folder, iteration, False)
    
    # The first call to execute_generated_code runs the initially generated code and captures the execution result.
    execution_result = execute_generated_code(current_output['response'], output_folder, iteration, False)
    print("execution_result - to see what is being added to the feedback collection")
    #print(execution_result)
    # Fetches feedback from previous iterations that are semantically similar to the current output.
    feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'] + execution_result)
    feedback_collection.add(
        ids=[f"iteration_{iteration}"],
        embeddings=[feedback_embedding['embedding']],
        documents=[current_output['response'] + "\n" + execution_result],
        metadatas=[{"iteration": iteration}]
    )
    
    # Retrieve relevant feedback from previous iterations
    query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'])
    
    # Ensure n_results is at least 1
    n_results = max(1, min(iteration, 7))
    relevant_feedback = feedback_collection.query( 
        query_embeddings=[query_embedding['embedding']],
        n_results=n_results
    )
    
    # Retrieve all Python files
    if python_files_collection.count() > 0:
        total_docs = python_files_collection.count()
        relevant_files = python_files_collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=total_docs  # Retrieve all documents
        )
        
        # Sort the results by relevance score (if available)
        if 'distances' in relevant_files:
            sorted_indices = sorted(range(len(relevant_files['distances'][0])), 
                                    key=lambda k: relevant_files['distances'][0][k])
            
            sorted_documents = [relevant_files['documents'][0][i] for i in sorted_indices]
            relevant_files['documents'] = [sorted_documents]
        
        # Limit the number of documents to include in the prompt if necessary
        max_docs_to_include = 2  # Adjust this number as needed
        relevant_files['documents'][0] = relevant_files['documents'][0][:max_docs_to_include]
    else:
        relevant_files = {"documents": ["No relevant Python files found."]}
    
    # Construct the refinement prompt with relevant feedback and Python files
    refinement_prompt = f"""
    IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
    DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
    You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:

    {current_output['response']}
    The code was executed with the following result:
    {execution_result}
    You must fix the results. I need the metaheuristic to run correctly. 
    Here is relevant feedback from previous iterations, DO NOT GENERATE THE SAME CODE GENERATED IN THE PREVIOUS ITERATIONS:
    {relevant_feedback['documents']}

    Here are relevant Python files that might be helpful:
    {relevant_files['documents']}

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
    sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
    import benchmark_func as bf
    import metaheuristic as mh


    fun = bf.HappyCat(2)
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
        (   # Search operator 2
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
        model=model,
        prompt=refinement_prompt
    )
    
    # Write refined output
    execute_generated_code(refined_output['response'], output_folder, iteration, False)
    
    return refined_output['response']


def self_refine_with_optuna(data, model, output_folder, iteration_number):
    logger.debug(f"Starting self_refine_with_optuna function for iteration {iteration_number}")
    
    # get or create the collection (here we are creating it)
    optuna_collecting = chromadb.Client().get_or_create_collection(name="optuna_collecting")
    #logger.debug("Retrieved optuna_collecting collection")
    
    print("data_optuna_is_important: ", data)
    # Uses Ollama to generate an initial response based on the given prompt and data.
    current_output_optuna = ollama.generate(
        model=model,
        prompt=f"Using this data: {data}. Respond to this prompt: {optuna_prompt}"
    )
     # Retrieve relevant feedback from previous iterations
    query_embedding_optuna = ollama.embeddings(model="mxbai-embed-large", prompt=current_output_optuna['response'])
    

    # Retrieve all Python files
    if optuna_collecting.count() > 0:
        total_docs = optuna_collecting.count()
        relevant_files = optuna_collecting.query(
            query_embeddings=[query_embedding_optuna['embedding']],
            n_results=total_docs  # Retrieve all documents
        )
        
        # Sort the results by relevance score (if available)
        if 'distances' in relevant_files:
            sorted_indices = sorted(range(len(relevant_files['distances'][0])), 
                                    key=lambda k: relevant_files['distances'][0][k])
            
            sorted_documents = [relevant_files['documents'][0][i] for i in sorted_indices]
            relevant_files['documents'] = [sorted_documents]
        
        # Limit the number of documents to include in the prompt if necessary
        max_docs_to_include = 2  # Adjust this number as needed
        relevant_files['documents'][0] = relevant_files['documents'][0][:max_docs_to_include]
    else:
        relevant_files = {"documents": ["No relevant Optuna files found."]}

    try:
        # Read the content of the execution_iteration_number.py file
        input_file_path = os.path.join(output_folder, f'execution_iteration_{iteration_number}.py')
        with open(input_file_path, 'r') as file:
            original_code = file.read()

        logger.debug(f"Generating Optuna-enhanced output for iteration {iteration_number}")

        full_prompt_with_optuna = f"""
You are a computer scientist specializing in natural computing and metaheuristic algorithms. You have been tasked with refining and improving the following output:
Enhance the following metaheuristic code by incorporating Optuna for hyperparameter tuning:
REMEMBER: 
1. EVERY EXPLANATION MUST START WITH '#'. 
2. DO NOT USE ANY MARKDOWN CODE BLOCKS such as ```python or ```
3. ONLY USE INFORMATION FROM THE optuna_builder folder (the one in the optuna_collection) and the information provided in this prompt.
4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
6. If you ever use genetic crossover, you must use genetic mutation as well. 
8. Checking for any logical errors or inconsistencies.
9. Improving the explanation and justification.

Please add Optuna to optimize the parameters of the GIVEN METAHEURISTIC. 
Ensure the Optuna-enhanced version still follows the original structure and logic.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.
DO NOT USE TRIPLE BACKTICKS (```) ANYWHERE IN YOUR RESPONSE. ALL OUTPUT MUST BE PLAIN TEXT.
USE THIS SAME CODE, do not create any other code: 
{original_code}

USE ONLY THIS DATA for reference on enhancing the previous code: {data}

FOLLOW EXACTLY the following template for the optuna-enhanced metaheuristic:
# Name: [Your chosen name for the optuna-enhanced metaheuristic]
# Code:

import optuna
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

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
            ... generated operators as needed
        ]

        fun = bf.HappyCat(2) # This is the selected problem, the problem may vary depending on the case.
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
"""
        current_output_with_optuna = ollama.generate(
            model=model,
            prompt=full_prompt_with_optuna
        )
        logger.debug(f"Full prompt for Ollama: {full_prompt_with_optuna}")
        
        logger.info(f"Optuna-enhanced output generated for iteration {iteration_number}")
        
        execute_generated_code(current_output_with_optuna['response'], output_folder, iteration_number, True)
        
        #optuna_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output_with_optuna['response'] + execution_result_with_optuna)
        #optuna_collecting.add(
        #    ids=[f"optuna_refinement_{iteration_number}"],
        #    embeddings=[optuna_embedding['embedding']],
        #    documents=[current_output_with_optuna['response'] + "\n" + execution_result_with_optuna],
        #    metadatas=[{"refinement": f"optuna_{iteration_number}"}]
        #)
        
        logger.debug(f"Completed self_refine_with_optuna function for iteration {iteration_number}")
        return current_output_with_optuna['response']
    
    except Exception as e:
        logger.error(f"Error in self_refine_with_optuna for iteration {iteration_number}: {str(e)}")
        raise

def execute_generated_code(code, output_folder, iteration, is_optuna):
    print("sitieneoptuna") if is_optuna else print("NOOOOOOOOONE")
    prefix = "execution_optuna_" if is_optuna else "execution_"
    # os.path.join()`: This function is used to create a proper file path string 
    # that works across different operating systems.
    file_name =  os.path.join(output_folder, f'{prefix}iteration_{iteration}.py')
    #file_path = os.path.join(output_folder,  f'{prefix}{file_name}.py')
    #print(file_path)
    with open(file_name, 'w') as f:
        f.write(code)
    
    try:
        result = subprocess.run(['python', file_name], capture_output=True, text=True, timeout=30)
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

# In your main execution:
if __name__ == "__main__":
    logger.debug("Starting main execution")
    try:
        # Your existing code to generate the output
        logger.debug("Generating embeddings for main prompt")
        response = ollama.embeddings(
            prompt=prompt,
            model="mxbai-embed-large"
        )
        logger.debug("Querying main collection")
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1
        )
        data = results['documents'][0][0]
        # New code to query optuna_collection
        logger.debug("Generating embeddings for Optuna-related query")
        optuna_response = ollama.embeddings(
            prompt=optuna_prompt,
            model="mxbai-embed-large"
        )
        logger.debug("Querying Optuna collection")
        optuna_results = optuna_collection.query(
            query_embeddings=[optuna_response["embedding"]],
            n_results=1
        )
        optuna_data = optuna_results['documents'][0][0]

        # Combine the data
        #combined_data = f"{data}\n\nOptuna-related information:\n{optuna_data}"

        # Create output folder
        logger.debug("Creating output folder")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(current_dir, f'ollama_output_{timestamp}')
        os.makedirs(output_folder)
        
        # Generate and refine the original output
        logger.debug("Starting________________")
        #  BEFORE: original_refined_output = self_refine(prompt, combined_data, "deepseek-coder-v2", output_folder)

        max_iterations = 7
        for i in range(max_iterations):
            logger.debug(f"Starting refinement iteration {i}")
            refined_output = self_refine(prompt, data, "deepseek-coder-v2", output_folder, i)
            logger.info(f"Refined output for iteration {i} generated")

            logger.debug(f"Starting Optuna refinement for iteration {i}")
            optuna_refined_output = self_refine_with_optuna(optuna_data, "codegemma", output_folder, i)
            client.delete_collection(name="feedback_collection")
            #client.delete_collection(name="collection")
            #client.delete_collection(name="optuna_builder")
            client.delete_collection(name="algorithm_creation")


        logger.debug("Main execution completed")
        client.delete_collection(name="optuna_builder")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")
        logger.exception("Exception details:")
        raise














