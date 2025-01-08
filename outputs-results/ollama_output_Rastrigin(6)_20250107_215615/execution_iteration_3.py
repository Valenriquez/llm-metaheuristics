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
    (  # Search operator 1: Central Force Dynamic
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'all'
    ),
    (
        # Search operator 2: Random Flight
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'gaussian',
            'beta': 1.5
        },
        'probabilistic'
    ),
    (
        # Search operator 3: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # Uncomment if needed for debugging
# met.run() # Uncomment to run the metaheuristic

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
# This hybrid metaheuristic combines the strengths of three different search operators: 
# Central Force Dynamic, Random Flight, and Local Random Walk. The Central Force Dynamic operator helps in exploring 
# the solution space efficiently, while the Random Flight operator aids in fine-tuning near optimal solutions. The Local Random Walk 
# ensures thorough exploration and avoiding local optima. By integrating these operators, this hybrid approach aims to balance exploration and exploitation 
# effectively, leading to better and more robust results on the Rastrigin benchmark function.