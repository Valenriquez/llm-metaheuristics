# Name: Hybrid Adaptive Swarm Optimization with Random Sampling (HASORS)

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
        'swarm_dynamic',
        {
            'factor': 0.5511270777623494,
            'self_conf': 2.7642509050645887,
            'swarm_conf': 2.7167957387913004,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
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
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Adaptive Swarm Optimization with Random Sampling (HASORS) combines the strengths of swarm optimization and random sampling. 
# It uses a dynamic swarm operator to guide the search process, while also incorporating randomness through random sampling to explore uncharted areas.
# This hybrid approach aims to balance exploration and exploitation, leading to more effective global optimization.