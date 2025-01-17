# Name: HybridRandomSearchSwarmDynamic

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
    (
        'random_search',
        {
            'scale': 0.63157840037779,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6974674051882006,
            'self_conf': 2.7480931119240735,
            'swarm_conf': 2.2231080541366084,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
# met.verbose = True
# met.run()

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
# This metaheuristic combines random search with swarm dynamic to explore the solution space efficiently.
# Random search is used to escape local minima by exploring the search space randomly, utilizing a scale of 0.63157840037779 and a uniform distribution.
# Swarm dynamic simulates the behavior of a swarm, allowing agents to communicate and coordinate their search efforts,
# enhancing exploration and exploitation capabilities with parameters such as factor=0.6974674051882006, self_conf=2.7480931119240735, swarm_conf=2.2231080541366084, and a constriction version for enhanced convergence.