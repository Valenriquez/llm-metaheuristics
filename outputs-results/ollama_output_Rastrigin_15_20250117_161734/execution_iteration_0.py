# Name: HybridMetaheuristic

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

heur = [
    ('random_search',
     {
         'scale': 0.15600035023019548,
         'distribution': 'uniform'
     },
     'greedy'),
    
    ('swarm_dynamic',
     {
         'factor': 0.9089947343003356,
         'self_conf': 2.018178285633897,
         'swarm_conf': 2.1069152398170363,
         'version': 'constriction',
         'distribution': 'gaussian'
     },
     'probabilistic')
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)

# Run the metaheuristic with the same problem 30 times
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines a random search operator with a swarm dynamic operator. The random search helps explore the solution space, while the swarm dynamic operator efficiently converges towards the optimal solution using the specified parameters. The combination aims to balance exploration and exploitation effectively.