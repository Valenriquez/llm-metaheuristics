# Name: Hybrid Metaheuristic

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
    ('central_force_dynamic',
     {
         'gravity': 0.001,
         'alpha': 0.01,
         'beta': 1.5,
         'dt': 1.0
     },
     'metropolis'),

    ('random_flight',
     {
         'scale': 1.0,
         'distribution': 'uniform',
         'beta': 1.5
     },
     'greedy'),

    ('spiral_dynamic',
     {
         'radius': 0.9,
         'angle': 22.5,
         'sigma': 0.1
     },
     'probabilistic'),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The hybrid metaheuristic combines the strengths of three different operators: Central Force Dynamic, Random Flight, and Spiral Dynamics. Each operator is chosen based on its suitability for handling different aspects of the optimization problem. The Central Force Dynamic operator helps in converging to a solution quickly, while the Random Flight ensures exploration and diversity. The Spiral Dynamics operator aids in fine-tuning the solution by simulating the behavior of a spiral motion.
# By using these operators with appropriate parameters and selectors, the hybrid approach aims to balance exploration and exploitation, leading to an efficient and robust optimization process.