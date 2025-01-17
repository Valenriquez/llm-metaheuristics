# Name: Adaptive Hybrid Metaheuristic (AHM)
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(6)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'metropolis'
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
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Adaptive Hybrid Metaheuristic (AHM) combines three different search operators to improve the exploration and exploitation capabilities of the metaheuristic. 
# 1. 'random_search' is used for global exploration with a small step size and a uniform distribution.
# 2. 'swarm_dynamic' incorporates social behaviors to facilitate faster convergence.
# 3. 'local_random_walk' enhances fine-tuning around promising solutions with a greedy approach.
# This hybrid strategy aims to balance exploration and exploitation effectively, leading to better performance on the Ackley1 benchmark function.