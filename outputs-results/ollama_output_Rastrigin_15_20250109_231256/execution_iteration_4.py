# Name: HarmonySearchWithSelectors

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
            'scale': 0.07805436758777234,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001725451039142446,
            'alpha': 0.09540482420198948,
            'beta': 3.8031934001867,
            'dt': 1.9665519567498992
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7028641523524904,
            'self_conf': 2.8137028621222924,
            'swarm_conf': 2.9460403934093584,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'random_sample',
        {},
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True
# met.run()

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This metaheuristic combines several search operators (random_search, central_force_dynamic, swarm_dynamic, random_sample) with different selectors (greedy, all, probabilistic, metropolis). The combination aims to leverage the strengths of various optimization techniques while managing their interactions through controlled selection processes. By running the algorithm multiple times and collecting final fitness values, we can analyze the robustness and performance of our metaheuristic approach.