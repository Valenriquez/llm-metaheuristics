# This code is used to create a metaheuristic for an optimization problem using optuna and ollama.
import optuna
import ollama
import chromadb
import numpy as np
import os
import benchmark_func as bf
import sys
import datetime
import subprocess
import time
import logging
from sklearn.neighbors import NearestNeighbors
import nltk
from mattsollamatools import chunker


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.append('llm-metaheuristics/algorithm_creation')

# Define the function
fun = bf.Ackley1(2)

# Get the function name
fun_name = fun.__class__.__name__

# Initialize ChromaDB client
client = chromadb.Client()

def read_python_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Directory containing Python files
python_files_dir = 'llm-metaheuristics/algorithm_creation'
optuna_files_dir = 'llm-metaheuristics/optuna_builder'


 # Create a collection for Python files
collection = client.create_collection(name="algorithm_creation")
optuna_collection = client.create_collection(name="optuna_builder")

# Process each Python file in the directory
for filename in os.listdir(python_files_dir):
    if filename.endswith('.py') or filename.endswith('.txt'):
        file_path = os.path.join(python_files_dir, filename)
        file_content = read_python_file(file_path)
        
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

# Process each Python file in the directory
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
1. Use only the function: bf.Rastrigin(2)
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


fun = bf.Rastrigin(2)
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
            (  
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

"""
 
def self_refine(initial_prompt, data, model, output_folder, max_iterations=7, python_files_dir='llm-metaheuristics/algorithm_creation'):
    # Initialize a collection for storing feedback
    feedback_collection = chromadb.Client().create_collection(name="feedback_collection")
    
    # Initialize a collection for Python files if not already done
    python_files_collection = chromadb.Client().get_or_create_collection(name="algorithm_creation")
    
    current_output = ollama.generate(
        model=model,
        prompt=f"Using this data: {data}. Respond to this prompt: {initial_prompt}"
    )
    
    print(current_output['response'])
    
    # Write initial output  # commenting to avoid to much files 
    #write_output_to_file(current_output['response'], output_folder, 0)
    
    for i in range(max_iterations):
        execution_result = execute_generated_code(current_output['response'], output_folder, i, is_optuna=False)
        
        # Add the current output and execution result to the feedback collection
        feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'] + execution_result)
        feedback_collection.add(
            ids=[f"iteration_{i}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[current_output['response'] + "\n" + execution_result],
            metadatas=[{"iteration": i}]
        )
        
        # Retrieve relevant feedback from previous iterations
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'])
        
        # Ensure n_results is at least 1
        n_results = max(1, min(i, 7))
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
        Here is relevant feedback from previous iterations:
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

        refined_output = ollama.generate(
            model=model,
            prompt=refinement_prompt
        )
        
        # Write refined output
        write_output_to_file(refined_output['response'], output_folder, i+1, is_optuna=False)
        
        # Check if the refinement made significant changes
        if refined_output['response'].strip() == current_output['response'].strip():
            print(f"No significant changes after iteration {i+1}. Stopping refinement.")
            break
        
        current_output = refined_output
        print(f"Completed refinement iteration {i+1}")
    
    return current_output['response']


def self_refine_with_optuna(original_code, data, model, output_folder, max_iterations=7):
    logger.debug("Starting self_refine_with_optuna function")

    optuna_collecting = chromadb.Client().create_collection(name="optuna_collecting")
    logger.debug("Retrieved optuna_builder collection")
    
    try:
        logger.debug("Generating initial Optuna-enhanced output")
        current_output = ollama.generate(
            model=model,
            prompt=f"""
            Enhance the following metaheuristic code by incorporating Optuna for hyperparameter tuning:
            Add a "hello" function at the end of the code that prints "hello".
            {original_code}
            
            Use this data for reference: {data}
            
            Please add Optuna to optimize the parameters of the metaheuristic. 
            Ensure the Optuna-enhanced version still follows the original structure and logic.
            Add a "hello" function at the end of the code that prints "hello".

            """
        )
        
        logger.info("Initial Optuna-enhanced output generated")
        print("Initial Optuna-enhanced output:")
        print(current_output['response'])
        
        # Write initial Optuna-enhanced output
        write_output_to_file(current_output['response'], output_folder, "initial_optuna", is_optuna=True)
        logger.debug("Wrote initial Optuna-enhanced output to file")
        
    except Exception as e:
        logger.error(f"Error generating initial Optuna-enhanced output: {str(e)}")
        logger.exception("Exception details:")
        return None

    for i in range(max_iterations):
        logger.debug(f"Starting iteration {i} of Optuna refinement")
        execution_result = execute_generated_code(current_output['response'], output_folder, i, is_optuna=True)
        logger.debug(f"Executed generated code for iteration {i}")
        
        feedback_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'] + execution_result)
        optuna_collecting.add(
            ids=[f"optuna_iteration_{i}"],
            embeddings=[feedback_embedding['embedding']],
            documents=[current_output['response'] + "\n" + execution_result],
            metadatas=[{"iteration": i}]
        )
        
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=current_output['response'])
        n_results = max(1, min(i, 7))
        relevant_feedback = optuna_collecting.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        
        refinement_prompt = f"""
        Refine the Optuna-enhanced metaheuristic:
        {current_output['response']}
        
        Execution result:
        {execution_result}
        
        Previous feedback:
        {relevant_feedback['documents']}
        
        Improve the Optuna integration and fix any issues. Ensure it's working correctly.
        """

        refined_output = ollama.generate(
            model=model,
            prompt=refinement_prompt
        )
        
        # Write refined Optuna output
        write_output_to_file(refined_output['response'], output_folder, f"optuna_iteration_{i+1}", is_optuna=True)
        
        if refined_output['response'].strip() == current_output['response'].strip():
            print(f"No significant changes after Optuna iteration {i+1}. Stopping refinement.")
            break
        
        current_output = refined_output
        print(f"Completed Optuna refinement iteration {i+1}")
    
    logger.debug("Completed self_refine_with_optuna function")
    return current_output['response']

def write_output_to_file(content, output_folder, filename, is_optuna):
    prefix = "execution_optuna_" if is_optuna else "execution_"
    file_path = os.path.join(output_folder, f'{prefix}{filename}.py')
    with open(file_path, 'w') as f:
        f.write(content)


def execute_generated_code(code, output_folder, iteration, is_optuna):
    prefix = "execution_optuna_" if is_optuna else "execution_"
    file_name = f'{prefix}iteration_{iteration}.py'
    file_path = os.path.join(output_folder, file_name)
    with open(file_path, 'w') as f:
        f.write(code)
    
    try:
        result = subprocess.run(['python', file_path], capture_output=True, text=True, timeout=30)
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
        logger.debug("Generating embeddings")
        response = ollama.embeddings(
            prompt=prompt,
            model="mxbai-embed-large"
        )
        logger.debug("Querying collection")
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1
        )
        data = results['documents'][0][0]

        # Create output folder
        logger.debug("Creating output folder")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(current_dir, f'ollama_output_{timestamp}')
        os.makedirs(output_folder)

        # Generate and refine the original output
        logger.debug("Starting self_refine for original output")
        original_refined_output = self_refine(prompt, data, "deepseek-coder-v2", output_folder)
        logger.info("Final refined output (without Optuna):")
        print(original_refined_output)
        write_output_to_file(original_refined_output, output_folder, "final_refined", is_optuna=False)
        execute_generated_code(original_refined_output, output_folder, "final_refined_execution", is_optuna=False)

        # Generate and refine the Optuna-enhanced version
        logger.debug("Starting self_refine_with_optuna")
        optuna_refined_output = self_refine_with_optuna(original_refined_output, data, "deepseek-coder-v2", output_folder)
        if optuna_refined_output is not None:
            logger.info("Final Optuna-enhanced output:")
            print(optuna_refined_output)
            write_output_to_file(optuna_refined_output, output_folder, "final_optuna_refined", is_optuna=True)
            execute_generated_code(optuna_refined_output, output_folder, "final_optuna_refined_execution", is_optuna=True)
        else:
            logger.error("Failed to generate Optuna-enhanced output")
            # Attempt to generate a basic Optuna integration
            basic_optuna_integration = f"""
            import optuna
            
            def objective(trial):
                # Add Optuna parameter suggestions here
                return 0  # Replace with actual objective calculation
            
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=100)
            
            print("Best trial:")
            print("  Value: ", study.best_value)
            print("  Params: ", study.best_params)
            
            {original_refined_output}
            
            def hello():
                print("hello")
            
            hello()
            """
            write_output_to_file(basic_optuna_integration, output_folder, "basic_optuna_integration", is_optuna=True)
            execute_generated_code(basic_optuna_integration, output_folder, "basic_optuna_integration_execution", is_optuna=True)

        logger.debug("Main execution completed")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")
        logger.exception("Exception details:")
        raise
