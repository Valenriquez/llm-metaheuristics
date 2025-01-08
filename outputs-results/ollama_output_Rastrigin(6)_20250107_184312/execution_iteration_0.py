# Name: Adaptive Hybrid Metaheuristic (AHM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)
prob = fun.get_formatted_problem()

# Define the hybrid search operators with parameters and selectors
heur = [
    (
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'version': 'inertial'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # Uncomment to enable verbose output
# met.run() # Uncomment to run the metaheuristic

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Adaptive Hybrid Metaheuristic (AHM) combines three different search operators: Random Search, Central Force Dynamic, and Spiral Dynamic. Each operator is configured with specific parameters to suit the Rastrigin benchmark function. The 'all' selector for Random Search ensures that it explores the entire solution space, 'greedy' for Central Force Dynamic encourages quick convergence to a suboptimal solution, and 'probabilistic' for Spiral Dynamic promotes exploration of the solution landscape in a probabilistic manner. This hybrid approach aims to balance exploration and exploitation, potentially leading to better solutions and more robust performance across multiple runs.