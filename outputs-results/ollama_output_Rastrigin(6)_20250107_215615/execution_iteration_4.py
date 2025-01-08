# Name: Hybrid Evolutionary Algorithm

# Code:

import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem
prob = fun.get_formatted_problem()

# Define operators and selectors with parameters based on the provided data
operators = [
    ('random_search', {'scale': 0.1, 'distribution': 'gaussian'}, 'greedy'),
    ('central_force_dynamic', {'gravity': 0.002, 'alpha': 0.02, 'beta': 1.6, 'dt': 1.5}, 'metropolis'),
    ('local_random_walk', {'probability': 0.7, 'scale': 1.2, 'distribution': 'uniform'}, 'inertial')
]

heur = [(operator, params, selector) for operator, params, selector in operators]

# Initialize and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []

# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=200, num_agents=150)
    met.reset_historicals()
    met.verbose = False
    met.run()

    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This Hybrid Evolutionary Algorithm combines three different search operators: random_search, central_force_dynamic, and local_random_walk. Each operator has specific parameters to control its behavior. The 'greedy' selector is used for the first two operators, while 'metropolis' is used for the third one to allow for probabilistic acceptance of worse solutions, which helps in escaping local optima.
# Running the metaheuristic 30 times allows us to analyze the stability and convergence properties of our hybrid approach.