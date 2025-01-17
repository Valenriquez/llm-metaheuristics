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
    (
        'random_search',
        {
            'scale': 1.8549271943957668,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5771164678150038,
            'self_conf': 2.7435612504977387,
            'swarm_conf': 2.8296740405407954,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6038503336793515,
            'angle': 5.974327746977677,
            'sigma': 0.27050427569427277
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
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
# The HybridMetaheuristic combines three different search operators: random_search, swarm_dynamic, and spiral_dynamic. This approach allows the algorithm to explore the solution space more effectively by utilizing the strengths of each operator. The random_search helps in exploring new areas, while swarm_dynamic and spiral_dynamic guide the search towards promising regions.
# By running the metaheuristic multiple times (30 repetitions), we can get a better understanding of its performance and stability across different starting points. This approach is particularly useful for problems like Rastrigin's function, which has many local minima, where a robust exploration strategy is crucial to finding the global optimum.