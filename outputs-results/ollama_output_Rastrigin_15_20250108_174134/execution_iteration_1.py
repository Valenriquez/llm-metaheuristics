# Name: Hybrid Metaheuristic with Operator Selection

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

# Initialize operators with varied parameters and selectors
operators = [
    ('random_search', {'scale': 0.01, 'distribution': 'uniform'}, 'greedy'),
    ('central_force_dynamic', {'gravity': 0.01, 'alpha': 0.01, 'beta': 1.5, 'dt': 1.0}, 'all'),
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform'}, 'probabilistic'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'gaussian'}, 'greedy')
]

heur = [
    (operator_name, operator_params, selector) for operator_name, operator_params, selector in operators
]

# Initialize metaheuristic with problem and selected operators
met = mh.Metaheuristic(prob, heur, num_iterations=1000)

# Run the metaheuristic multiple times to gather fitness data
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()

    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines four different operators (Random Search, Central Force Dynamic, Local Random Walk, and Swarm Dynamic) with varied parameter settings. Each operator is paired with a specific selector to manage the exploration-exploitation trade-off during the optimization process.
# The selection of these operators aims to leverage their unique strengths in handling different aspects of the problem space:
# - 'random_search' for broad exploration
# - 'central_force_dynamic' for attraction towards optimal solutions
# - 'local_random_walk' for fine-grained exploration around promising areas
# - 'swarm_dynamic' for collective intelligence and diversity in population
# The hybrid approach helps to balance the search process, potentially leading to more efficient convergence towards global optima.