# Name: Hybrid Adaptive Metaheuristic

# Code:

import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.009406961247945378,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.6533124327622927,
            'self_conf': 2.8092553938444698,
            'swarm_conf': 2.6683954753722747,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        # Search operator 3: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.7966669047164824,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=2000)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=2000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid adaptive metaheuristic combines the strengths of random search, swarm dynamic, and local random walk operators.
# The random search operator helps explore new regions with a smaller scale, while the swarm dynamic operator exploits promising areas by mimicking social behavior using constriction version with updated parameters.
# The local random walk operator ensures fine-grained exploration around the current best solution with an increased probability.
# By using a probabilistic selector for the swarm dynamic operator, we allow it to occasionally make exploratory moves even if they are not immediately improving the fitness.
# This combination should provide a good balance between exploration and exploitation, potentially leading to better performance on the given problem.