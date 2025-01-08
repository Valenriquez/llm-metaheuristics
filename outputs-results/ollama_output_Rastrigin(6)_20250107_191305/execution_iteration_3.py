# Name: Adaptive Hybrid Metaheuristic

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

# Define the search operators with parameters and selectors
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
            'beta': 1.5,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'levy'
        },
        'metropolis'
    )
]

# Create the Metaheuristic instance
met = mh.Metaheuristic(prob, heur, num_iterations=100)

# Initialise the fitness register
fitness = []

# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

# Convert fitness data to NumPy array
fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This metaheuristic combines three different search operators: Random Search, Central Force Dynamics, and Local Random Walk. Each operator is configured with specific parameters and a selector to control the exploration and exploitation phases of the optimization process.
#
# The `Random Search` operator uses a uniform distribution to explore the solution space randomly, which helps in avoiding local optima.
#
# The `Central Force Dynamics` operator simulates the interaction between particles in a system under the influence of attractive or repulsive forces. It is initialized with Gaussian distribution parameters and uses a probabilistic selector to guide the search towards promising areas of the solution space.
#
# The `Local Random Walk` operator uses a levy distribution for random movements, which allows for efficient exploration of the problem landscape by sometimes taking large steps.
#
# By combining these operators, the hybrid metaheuristic aims to leverage their strengths in exploring and exploiting the solution space, leading to improved convergence and robustness.