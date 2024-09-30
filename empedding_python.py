import ollama
import chromadb
import numpy as np
import os
import benchmark_func as bf
import sys
import datetime

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

 # Create a collection for Python files
collection = client.create_collection(name="algorithm_creation")

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

prompt = """
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.

You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the {fun} optimization problem using only the operators and selectors from the parameters_to_take.txt file.

INSTRUCTIONS:
1. Use only the function: bf.{fun_name}
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
  operator: "random_search": {
    "parameters": {
      "scale": 1.0 or 0.01,
      "distribution": "uniform" or "gaussian" or "levy"]
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "central_force_dynamic": {
    "parameters": {
      "gravity": 0.001,
      "alpha": 0.01,
      "beta": 1.5,
      "dt": 1.0
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "differential_mutation": {
    "parameters": {
      "expression": "rand" or "best" or "current" or  "current-to-best" or "rand-to-best" or "rand-to-best-and-current",
      "num_rands": 1,
      "factor": 1.0
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "firefly_dynamic": {
    "parameters": {
      "distribution": "uniform" or "gaussian" or "levy",
      "alpha": 1.0,
      "beta": 1.0,
      "gamma": 100.0
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "genetic_crossover": {
    "parameters": {
      "pairing": "rank" or "cost" or "random" or"tournament_2_100",
      "crossover": "single" or "two" or "uniform" or "blend" or "linear_0.5_0.5",
      "mating_pool_factor": 0.4
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "genetic_mutation": {
    "parameters": {
      "scale": 1.0,
      "elite_rate": 0.1,
      "mutation_rate": 0.25,
      "distribution": "uniform" or "gaussian" or "levy"
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "gravitational_search": {
    "parameters": {
      "gravity": 1.0,
      "alpha": 0.02
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "random_flight": {
    "parameters": {
      "scale": 1.0,
      "distribution": "levy" or "uniform" or"gaussian",
      "beta": 1.5
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "local_random_walk": {
    "parameters": {
      "probability": 0.75,
      "scale": 1.0,
      "distribution": "uniform" or "gaussian" or "levy"
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "random_sample": {
    "parameters": {},
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "spiral_dynamic": {
    "parameters": {
      "radius": 0.9,
      "angle": 22.5,
      "sigma": 0.1
    },
    selector: "greedy" or "all" or"metropolis" or"probabilistic"
  },
  operator: "swarm_dynamic": {
    "parameters": {
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


fun = bf.{fun_name}
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    '[operator_name]',
    {{
         'parameter1': value1,
         'parameter2': value2,
         # ... more parameters as needed
    }},
    '[selector_name]'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))


# Short explanation and justification:
# [Your explanation here, each line starting with '#']

REMEMBER: 
1. EVERY EXPLANATION MUST START WITH '#'. 
2. DO NOT USE ANY MARKDOWN SYNTAX OR CODE BLOCKS. 
3. ONLY USE INFORMATION FROM THE parameters_to_take.txt FILE.
4. DO NOT INCLUDE ANY COMMENTS IN THE CODE SECTION.
5. ENSURE ALL PARAMETER NAMES AND VALUES APPEAR IN parameters_to_take.txt.
5. If you ever use genetic crossover, you must use genetic mutation as well. 

"""
 
# Your existing code to generate the output
response = ollama.embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

output = ollama.generate(
  model="deepseek-coder-v2",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Generate a unique folder name using a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(current_dir, f'ollama_output_{timestamp}')

# Create the new folder
try:
    os.makedirs(output_folder)
    print(f"Created new folder: {output_folder}")
except Exception as e:
    print(f"An error occurred while creating the folder: {e}")
    exit(1)  # Exit if we can't create the folder

# Define the file path inside the new folder
output_file_path = os.path.join(output_folder, 'ollama_output.py')

# Write the output to the file with error handling
try:
    with open(output_file_path, 'w') as file:
        file.write(output['response'])
    # Print a confirmation message
    print(f"Output has been written to {output_file_path}")
except Exception as e:
    print(f"An error occurred while writing the file: {e}")

# Print the output to the console
print("Generated output:")
print(output['response'])