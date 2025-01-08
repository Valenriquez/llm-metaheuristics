# Name: HybridMetaheuristic
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

heur = [
    (
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.005,
            'alpha': 0.02,
            'beta': 1.4,
            'dt': 0.9
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.55,
            'swarm_conf': 2.57,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 23.0,
            'sigma': 0.15
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=150, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HybridMetaheuristic combines four different metaheuristic operators to enhance the search process. 
# The random_search operator helps in exploring the solution space, central_force_dynamic encourages movement towards better solutions,
# swarm_dynamic simulates social behaviors for collective optimization, and spiral_dynamic introduces a dynamic pattern for exploration.
# The use of various selectors ensures that the best solutions are retained during the evolution process.
# This hybrid approach aims to balance exploration and exploitation, potentially leading to better performance on complex benchmark functions like Rastrigin.