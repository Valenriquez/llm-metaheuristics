# Name: Hybrid Adaptive Metaheuristic (HAM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HAM combines three different operators: swarm_dynamic, spiral_dynamic, and random_sample.
# The swarm_dynamic operator encourages collective behavior among agents,
# the spiral_dynamic operator guides agents in a spiral motion towards better solutions,
# and the random_sample operator allows for exploration of the solution space by sampling randomly. 
# This hybrid approach aims to balance exploration and exploitation while adapting dynamically during the search process.

# Addressed error: Removed unexpected indentation in the loop body.