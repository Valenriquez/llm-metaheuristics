# Name: Custom Metaheuristic for Benchmark Function Optimization

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(6)
prob = fun.get_formatted_problem()

# Define the search operators with their parameters and selectors
heur = [
    (
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'greedy'
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
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'levy'
        },
        'metropolis'
    )
]

# Initialize the Metaheuristic object
met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)

# Run the metaheuristic with the same problem 30 times and store the results
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
# The Custom Metaheuristic combines three different search operators: random search, swarm dynamic, and local random walk. 
# Each operator is configured with specific parameters to suit the Ackley1 benchmark function.
# The `greedy` selector for random search ensures that improvements are immediately accepted.
# The `probabilistic` selector for swarm dynamic allows for exploration by probabilistically accepting worse solutions.
# The `metropolis` selector for local random walk incorporates a simulated annealing approach to handle local minima.
# Running the metaheuristic 30 times helps in obtaining a more robust estimate of the solution quality.