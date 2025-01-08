# Name: Hybrid Metaheuristic Algorithm

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.005,
            'alpha': 0.02,
            'beta': 1.8
        },
        'metropolis'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.7,
            'scale': 1.2,
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic algorithm combines three different search operators: Random Search, Central Force Dynamics, and Local Random Walk. The parameters have been tuned to balance exploration and exploitation effectively.
# The Random Search operator is used for initial exploration with a Gaussian distribution and a small scale.
# The Central Force Dynamics operator incorporates social forces similar to physics simulations, enhancing the algorithm's ability to navigate complex landscapes.
# The Local Random Walk operator ensures detailed exploration around promising regions with a uniform distribution.
# By combining these strategies, the hybrid approach aims to find better solutions more efficiently compared to using a single method.