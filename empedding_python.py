import ollama
import chromadb
import numpy as np
import os
import benchmark_func as bf
import sys
import datetime


sys.path.append('llm-metaheuristics/algorithm_creation')

fun = bf.Rastrigin(3)

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

# an example prompt
prompt = """ DO NOT WRITE ```python" or "```"  or something similar in the response,
You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel 
metaheuristic algorithms to solve optimization problems, the purpose is to design novel metaheuristic algorithms to solve optimization problems, but by trying the default.txt operators and selectors to find out which is the best one suited for the given function.

These are the following instructions: 

Using the function: {fun} and respecting the given operators and selectors combination from parameters_to_take.txt file, 
example: 

('differential_mutation', {'expression': 'current-to-best', 'num_rands': 1, 'factor': 1.0}, 'greedy')

each of the metaheuristic optimization function has a search space between -1.0 (lower bound) and 1.0 (upper bound). The dimensionality can be varied.
The number of iterations "num_iterations" must be 100 and there must not be more than two search operators (add__operator__), remember that every operator must have its own selector (add__selector__).
ONLY USE THE SEARCH OPERATORS AND SELECTORS FROM THE parameters_to_take.txt file.

An example of such code is as follows:

DO NOT WRITE ```python" or "```"  or something similar in the response, rather give the response in the format:
## Name: <name>  (make sure to comment the name of the metaheuristic)
## Code: <code> 

## Import all the needed modules (you must also import the benchmark function)
import <module>


heur = [( # Search operator 1
    'differential_mutation',  # Perturbator   
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 1,
        'factor': 1.0},
    'greedy'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

## Explanation of the code: (make sure to comment the explanation after the code)
COMMENT EVER TEXT AFTER THE WRITEN CODE. 
DO NOT WRITE ```python" or "```"  or something similar in the response,
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
  model="codegemma",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

# Print the response to the console
print(output['response'])

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Generate a unique filename using a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file_path = os.path.join(current_dir, f'ollama_output_{timestamp}.py')

# Write the output to the file with error handling
try:
    with open(output_file_path, 'w') as file:
        file.write(output['response'])
    # Print a confirmation message
    print(f"Output has been written to {output_file_path}")
except Exception as e:
    print(f"An error occurred while writing the file: {e}")

# If you still want to print the output to the console as well
print(output['response'])