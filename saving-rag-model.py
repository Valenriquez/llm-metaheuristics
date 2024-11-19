
"""
Simple RAG example with ollama

"""


import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
print("Current working directory:", os.getcwd())

# open a file and return paragraphs
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def save_embeddings(filename, embeddings):
    # create dir if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings


# find cosine similarity of every chunk to a given embedding
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def retrieve_context(needle, haystack, top_n=25):
    needle_norm = np.linalg.norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * np.linalg.norm(item)) for item in haystack
    ]
    ranked_chunks = sorted(zip(similarity_scores, haystack), reverse=True)
    return [chunk for _, chunk in ranked_chunks[:top_n]]



def main():
    SYSTEM_PROMPT = """You are a highly skilled computer scientist in the field of natural computing. Your task is to design a metaheuristic algorithm, 
        you should only use the information that was provided to you. Do not invent any new operators. 
        Remember that when writing the operator's names, they should be ALL in LOWER CASE AND WITH A '_' 
        instead of typing a space. Remember that, if the dimension is 3 or bigger, you should use a bigger selector, as there is more space to cover.
        Please in the 'fun' variable you must change it too: 'fun = bf.Ackley1(2)', do not change these values given. 
        In case there was an error, please fix it.

        When creating a metaheuristic take into account: You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response, 
        all outputs must be plain text. Use only the benchmark_function and its' dimension provided. 
        
        Format your response exaclty as follows.  
        Do not write anything before this format: 
            
        # Name: [Your chosen name for the metaheuristic]
        # Code:
        import sys
        from pathlib import Path

        project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
        sys.path.insert(0, str(project_dir))
        import benchmark_func as bf
        import metaheuristic as mh

        fun = bf.{self.benchmark_function}({self.dimensions}) # This is the selected problem, the problem may vary depending on the case.
        prob = fun.get_formatted_problem()

        heur = [
            (  # Search operator 1
                '[operator_name]',
                {
                    'parameter1': value1,
                    'parameter2': value2,
                    more parameters as needed
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

        print('x_best = {}, f_best = {}'.format(*met.get_solution()))

        # Initialise the fitness register
        fitness = []
        # Run the metaheuristic with the same problem 30 times
        for rep in range(30):
            met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
            met.reset_historicals()
            met.verbose = False
            met.run()
            print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
            
            fitness.append(met.historical['fitness'])
            
        # Short explanation and justification:
        # [Your explanation here, each line starting with '#']

    """
    TEMPERATURE = 0
    # open file
    
    filename = "parameters_to_take.txt"
    paragraphs = parse_file(filename)

    embeddings = get_embeddings(filename, "mxbai-embed-large", paragraphs)

    prompt = """Your task is to design a metaheuristic algorithm, the 'fun' variable you must change it too: 'fun = bf.Ackley1(2)', do not change these values given.   
        When creating a metaheuristic take into account: You should NOT use any markdown code or use the triple backticks  (```) anywhere in your response, 
        all outputs must be plain text. Use only the benchmark_function and its' dimension provided. 
        
        Format your response exaclty as follows.  
        Do not write anything before this format: 
            
        # Name: [Your chosen name for the metaheuristic]
        # Code:
        import sys
        from pathlib import Path

        project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
        sys.path.insert(0, str(project_dir))
        import benchmark_func as bf
        import metaheuristic as mh

        fun = bf.{self.benchmark_function}({self.dimensions}) # This is the selected problem, the problem may vary depending on the case.
        prob = fun.get_formatted_problem()

        heur = [
            (  # Search operator 1
                '[operator_name]',
                {
                    'parameter1': value1,
                    'parameter2': value2,
                    more parameters as needed
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

        print('x_best = {}, f_best = {}'.format(*met.get_solution()))

        # Initialise the fitness register
        fitness = []
        # Run the metaheuristic with the same problem 30 times
        for rep in range(30):
            met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
            met.reset_historicals()
            met.verbose = False
            met.run()
            print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
            
            fitness.append(met.historical['fitness'])
            
        # Short explanation and justification:
        # [Your explanation here, each line starting with '#']

    
    """ 
    # strongly recommended that all embeddings are generated by the same model (don't mix and match)
    prompt_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)["embedding"]
    most_similar_chunks = retrieve_context(prompt_embedding, embeddings)[:25]

    response = ollama.chat(
        model="myqwen2.5",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(paragraphs[int(item[1])] for item in most_similar_chunks)
            },
            {"role": "user", "content": prompt},
        ],
    )

    print("\n\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()