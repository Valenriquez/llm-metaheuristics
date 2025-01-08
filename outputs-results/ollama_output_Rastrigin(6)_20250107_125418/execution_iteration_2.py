# Name: Adaptive Hybrid Metaheuristic (AHM)

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem.
prob = fun.get_formatted_problem()

# Define the search operators and selectors with updated parameters
heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.6821437509280424,
            'scale': 1.2366672134479393,
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8992690971488672,
            'self_conf': 2.8629493351632584,
            'swarm_conf': 2.9155147736618745,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8535498132024881,
            'angle': 22.729944451150743,
            'sigma': 0.24586575282950143
        },
        'greedy'
    )
]

# Initialize the metaheuristic and run it
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
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
# AHM combines several search operators (random_sample, local_random_walk, swarm_dynamic, spiral_dynamic) to explore the solution space.
# Each operator is selected with a specific selector: greedy, probabilistic, metropolis, and greedy respectively. 
# The combination of these operators allows for a diverse exploration strategy that can help in finding better solutions to the problem.