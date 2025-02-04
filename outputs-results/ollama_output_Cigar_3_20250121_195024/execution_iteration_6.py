# Name: HybridRandomizedMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(3)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Sample
        'random_sample',
        {},
        'greedy'
    ),
    (
        # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.5815380818766118,
            'self_conf': 2.647159926967265,
            'swarm_conf': 2.0680176651730795,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        # Search operator 3: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.7538632923493069,
            'scale': 0.26288537186497063,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The HybridRandomizedMetaheuristic combines three different search operators: Random Sample, Swarm Dynamic, and Local Random Walk. Each operator is designed to explore the solution space differently. 
# - 'random_sample' allows for an initial exploration of the problem space.
# - 'swarm_dynamic' mimics a social behavior where agents (solutions) influence each other's movement, promoting exploitation with the specified parameters that enhance its effectiveness.
# - 'local_random_walk' helps in fine-tuning and avoiding getting stuck in local optima by making small random steps with the provided parameters.
# This hybrid approach aims to balance exploration and exploitation, potentially leading to better solutions for the Cigar benchmark function.