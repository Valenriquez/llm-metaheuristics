import ollama
import chromadb
import numpy as np
import os
import benchmark_func as bf
import sys

sys.path.append('llm-metaheuristics/algorithm_creation')

fun = bf.Rastrigin(2)

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
    if filename.endswith('.py'):
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

# an example prompt
prompt = """You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel 
metaheuristic algorithms to solve optimization problems, I am currently using the following function: {fun}

Each of the metaheuristic optimization function has a search space between -1.0 (lower bound) and 1.0 (upper bound). The dimensionality can be varied.
An example of such code is as follows:

heur = [( # Search operator 1
    'differential_mutation',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 2,
        'factor': 1.0},
    'greedy'  # Selector
), (  # Search operator 2
    'differential_crossover',  # Perturbator
    {  # Parameters
        'crossover_rate': 0.2,
        'version': 'binomial'
    },
    'greedy'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <name>
# Code: <code>
# Import all the needed modules (you must also import the benchmark function)
import <module>



Create a Metaheuristic based on the given information in the algorithm_creation files. 
The template is given in the metaheuristic_selection.py file. You must ONLY USE the add__operator__ and add__selector__ that are given in the metaheuristic_selection.py file. 
In the given example code,  the "differential_mutation" is the perturbator, the "greedy" is the selector.
Which is the best selector and operator, use the given template to build the metaheuristic and run it. 

The number of iterations must be 100.
When given Explanation: or Benefits:, you must write all that text explanation and benefits text of the metaheuristic algorithm in a commented format.
""" 
 
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
  model="codegemma",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the file path where you want to save the output
output_file_path = os.path.join(current_dir, 'ollama_output.py')

# Write the output to the file
with open(output_file_path, 'w') as file:
    file.write(output['response'])

# Optionally, print a confirmation message
print(f"Output has been written to {output_file_path}")

# If you still want to print the output to the console as well
print(output['response'])