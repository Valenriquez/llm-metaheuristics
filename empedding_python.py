import ollama
import chromadb
import numpy as np
import os
import benchmark_func as bf
import sys
import datetime


sys.path.append('llm-metaheuristics/algorithm_creation')

fun = bf.Ackley1(2)

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
prompt = """
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCKS. ALL OUTPUT MUST BE PLAIN TEXT.

You are a computer scientist specializing in natural computing and metaheuristic algorithms. Your task is to design a novel metaheuristic algorithm for the {fun} optimization problem using only the operators and selectors from the parameters_to_take.txt file.

INSTRUCTIONS:
1. Use only the function: bf.{fun}
2. Use only operators and selectors from parameters_to_take.txt
3. The search space is between -1.0 (lower bound) and 1.0 (upper bound)
4. Set num_iterations to 100
5. Use no more than two search operators
6. Each operator must have its own selector
7. Fill all parameters for the chosen operator with your best recommendations
8. Create only one metaheuristic per response

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

# Name: [Your chosen name for the metaheuristic]
# Code:
 [All necessary import statements]
 fun = bf.{fun}
 prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    '[operator_name]',
    {{  # Parameters
         '[param1]': [value1],
         '[param2]': [value2],
         # ... more parameters as needed
     }},
     '[selector_name]'
 ),
 ( # Search operator 2 (if used)
     '[operator_name]',
     {{  # Parameters
         '[param1]': [value1],
         '[param2]': [value2],
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

REMEMBER: EVERY EXPLANATIONS, MUST START WITH '#'. DO NOT USE ANY MARKDOWN SYNTAX OR CODE BLOCKS.
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