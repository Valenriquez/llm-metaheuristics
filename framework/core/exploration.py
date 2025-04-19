import ollama 
import os
from dataclasses import dataclass
from chromadb.api.models.Collection import Collection

"""
Exploration:  Will create a metaheuristic and search for new creations. After the first iteration it will look on the feedback for inspiration on better 
metaheuirsics creation. 
- Will have access to feedback: WHOLE RESULTS FOLDER
- Will get hyperparameters for the refinement 
"""

@dataclass
class Exploration:
    problem_id: int
    dimensions: int
    num_of_agents: int 
    model: str = "qwen2.5-coder:latest"
    model_embed: str = "all-minilm:latest"
    feedback_collection:  Collection = None 
    operators_collection:  Collection = None 
    metaheuristic_template_collection:  Collection = None 
    made_metaheuristic: str = None

    def exploration(self, number_iteration):
        print("Beginning with exploration number --->",  number_iteration)
        output_response = None

        # Query for the Operators - - - - - - - - - - - - - - - - - - -
        output_operators = ollama.embeddings(
        prompt=f"Remember to write the operators' names",
        model=self.model_embed
        )
        results = self.operators_collection.query(
        query_embeddings=[output_operators["embedding"]],
        n_results=1
        )
        operators_data  = results['documents'][0][0]
        print("operators data:  ", operators_data)
        
        # Query for the Operators - - - - - - - - - - - - - - - - - - -
     
          # there won´t be any feedback else:
      
        # Query for the Feedback - - - - - - - - (asks for the metaheuristics and feedback) - - - - - - - - - - -
        output_feedback = ollama.embeddings(
        prompt=f"Give me the best operators for the given dimensions: {self.dimensions}",
        model=self.model_embed
        )
        results = self.operators_collection.query(
        query_embeddings=[output_feedback["embedding"]],
        n_results=1
        )
        #answer = results['documents'][0][0]
        #print("The best operators for the given problem---", answer)


        # Query for the Simple Metaheuristic Generation  - - - - - - - - - - - - - - - - - - -
        output = ollama.generate(
            model=self.model,
            prompt = f"""
                You are a highly skilled computer scientist in the field of natural computing. Your task is to design a metaheuristic algorithm, 
                you should only use the information that was provided to you. 
                Remember that when writing the operator's names, they should be ALL in LOWER CASE AND WITH A '_' instead of typing a space. 
                
                ### IMPORTANT INSTRUCTIONS:
                - DO NOT PROVIDE ANY TEXT OR EXPLANATION or </think>, ONLY CODE:
                1. **Code Format**: Avoid using triple backticks (` ``` `), Python-specific syntax, or markdown in the response.
                2. **Operator Diversity**: Ensure extracted operators reflect a wide range of strategies. Avoid limiting to only 2-3 operator types; include variety based on the data's complexity.
                3. **Operator Limit**: Do not include more than **4 operators** in the response.

                ### DATA USAGE RULES:
                - Do **not** modify operators, parameters, variables, or selectors from the provided data.
                - Strictly adhere to the provided details without inventing, omitting, or altering any information.

                ### OBJECTIVE:
                Incorporate the provided feedback to design a new metaheuristic approach. Aim to create a method that maintains or improves performance while reducing computational cost.

                ### PROVIDED DATA:
                {operators_data} 
                
                ### RESPONSE FORMAT:
                Please structure your response using the following template:
                heur = [
                    (  # Search Operator 1
                        '[operator_name]',
                        {{
                            'parameter1': value1,
                            'parameter2': value2,
                            # Add more parameters as necessary
                        }},
                        '[selector_name]'
                    ),
                    (  # Search Operator 2
                        '[operator_name]',
                        {{
                            'parameter1': value1,
                            'parameter2': value2,
                            # Add more parameters as necessary
                        }},
                        '[selector_name]'
                    )
                    # Add up to 4 operators total
                ]
            """
            )
        self.made_metaheuristic = output['response']
        # Query for the Simple Metaheuristic Generation  - - - - - - - - - - - - - - - - - - -

        # Query for the Metaheuristic Generation Complete File - - - - - - - - - - - - - - - - - - -
        template_file= f"""   
# This is the Python Iteration: {number_iteration}
# Author: [Your Name]
# Date: [Insert Date]

# Code:
import sys
from pathlib import Path
import ioh
project_dir = Path(__file__).resolve().parents[4] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[4]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from P1 import P1

def evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas):
        
    ioh_problem = P1.create_ioh_problem(problem_id, instance, dimension)
    fun = P1(variable_num=dimension, problem=ioh_problem)
    prob = fun.get_formatted_problem()

    def run_metaheuristic():
        met = mh.Metaheuristic(prob, heur, num_agents, num_iterations)
        met.verbose = False
        met.run()
        best_position, f_best = met.get_solution()
        return f_best, best_position

    # Ejecutar en paralelo el número de réplicas
    num_cores = min(multiprocessing.cpu_count(), num_replicas)
    results_parallel = Parallel(n_jobs=num_cores, prefer="threads")(delayed(run_metaheuristic)() for _ in range(num_replicas))

    # Extraer los valores de fitness de los resultados y calcular la métrica de rendimiento
    fitness_values = [result[0] for result in results_parallel]
    positions = [result[1] for result in results_parallel]
    fitness_median = np.median(fitness_values)
    iqr = np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)
    performance_metric = fitness_median + iqr

    # Fitness finales
    fitness_array = np.array(fitness_values).T
    print("final_fitness_array", fitness_array)

    # Retorna el mejor valor y la mejor posición encontrada en todas las réplicas
    best_fitness_index = np.argmin(fitness_values)
    best_position = positions[best_fitness_index]
    return performance_metric, best_position, fitness_array

{self.made_metaheuristic}    

problem_id = {self.problem_id}    
instance = 1
dimension = {self.dimensions}     
num_agents= {self.num_of_agents}    
num_iterations = 100
num_replicas = 100

performance_metric, best_position, fitness_array = evaluate_sequence_IOH(heur, problem_id, instance, dimension, num_agents, num_iterations, num_replicas)
print("Métrica de rendimiento (Mediana + IQR):", performance_metric)
print("Mejor posición encontrada:", best_position)

# Obtener y comparar con el óptimo
problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
optimal_fitness = problem.optimum.y                 
            """
        
        return template_file

            
