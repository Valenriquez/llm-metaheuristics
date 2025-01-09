# Name: Randomized Search with Dynamic Factors

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'greedy'),
    ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'metropolis'),
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform'}, 'probabilistic'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'gaussian'}, 'greedy')
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
# met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines several search operators to explore the solution space more effectively. The `random_sample` operator provides a baseline of randomness, while the `spiral_dynamic`, `local_random_walk`, and `swarm_dynamic` operators introduce dynamic factors that adapt to the problem's characteristics, potentially leading to better convergence or exploration strategies.
# Each operator is selected with a specific selector to manage the search process efficiently. The use of multiple operators allows the algorithm to handle different aspects of the optimization problem, increasing its robustness and performance.