# Name: Hybrid Metaheuristic for Rastrigin Function
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
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # Uncomment to see more detailed output
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
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines three different search operators: random_sample, swarm_dynamic, and local_random_walk. 
# The 'random_sample' operator provides an initial population of solutions, while 'swarm_dynamic' simulates the behavior of a swarm to explore the search space.
# 'local_random_walk' allows for fine-grained exploration around promising regions. The probabilistic selector for 'swarm_dynamic' and metropolis for 'local_random_walk' help in balancing exploitation and exploration.
# Running multiple iterations helps in stabilizing the solution quality, and collecting data from different runs provides insights into the performance of the metaheuristic on the Rastrigin function.