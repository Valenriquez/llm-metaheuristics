# Name: Enhanced Randomized Swarm Optimization (ERSO)

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 1.9732714012813914,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5461593697741652,
            'self_conf': 2.1654786359128715,
            'swarm_conf': 2.230668613100059,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6267916986546668,
            'angle': 23.17126990646491,
            'sigma': 0.09578488717547422
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
# met.verbose = True # please comment this line
# met.run() # please comment this line

fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic, ERSO, combines three different search operators: Random Search, Swarm Dynamic, and Spiral Dynamic. The combination aims to leverage the exploration capabilities of Random Search with the exploitation power of both Swarm and Spiral dynamics. Each operator is configured with default parameters that are known to work well for the Rastrigin function. The metaheuristic runs for 1000 iterations and repeats this process 30 times to gather fitness data, which helps in evaluating the robustness and effectiveness of the approach on the given benchmark problem.