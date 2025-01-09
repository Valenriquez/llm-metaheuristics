# Name: Hybrid Metaheuristic for Optimization Problems

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

# Define the search operators and their parameters with selectors
heur = [
    (
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
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
    )
]

# Initialize the metaheuristic with the problem and operators
met = mh.Metaheuristic(prob, heur, num_iterations=100)

# Run the metaheuristic multiple times to gather fitness data
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

# Convert the fitness data to a numpy array for analysis
fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])

print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic combines four different search operators: random search, central force dynamics, swarm optimization, and spiral dynamics. Each operator is configured with specific parameters to explore the solution space effectively.
# The hybrid approach aims to leverage the strengths of each operator in different stages of the search process, potentially leading to better convergence and more robust solutions for complex benchmark functions like Rastrigin.