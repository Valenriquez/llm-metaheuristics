# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.5054080298427472,
            'self_conf': 2.3178176813863747,
            'swarm_conf': 2.2468448853417327,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'random_sample',
        {},
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The HybridMetaheuristic combines the Swarm Dynamic operator with a Random Sample operator. 
# The Swarm Dynamic operator is effective for exploration, while the Random Sample operator helps in fine-tuning the solution.
# This hybrid approach aims to balance exploration and exploitation, potentially leading to better convergence properties on the Rastrigin function.

# If you encounter an error, address it as follows:
# Traceback (most recent call last):
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/outputs-results/ollama_output_Rastrigin_6_20250116_124115/execution_iteration_0.py", line 59, in <module>
#     met.run()
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 144, in run
#     self.apply_search_operator(perturbator, selector)
#   File "/Users/valeriaenriquezlimon/Documents/meta-h-generator/llm-metaheuristics/metaheuristic.py", line 100, in apply_search_operator
#     exec('Operators.' + operator_name + '(self.pop,' + operator_params)
#   File "<string>", line 1, in <module>
# TypeError: genetic_crossover() got an unexpected keyword argument 'self_conf'
# This error indicates that the genetic_crossover operator is being called with an incorrect parameter. 
# Ensure that the parameters for each operator are correctly defined and compatible with the operator's implementation., modify it in order to put these parameters.