# Name: Randomized Local Search with Modified Spiral Dynamics and Swarm Intelligence

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
        'random_sample',
        {},
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7875886887518272,
            'angle': 19.32684384731014,
            'sigma': 0.13312542635062263
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8425106536803727,
            'self_conf': 2.6318327293618946,
            'swarm_conf': 2.8899256869436094,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)
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
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This metaheuristic combines random sampling, modified spiral dynamics, and swarm intelligence to explore the search space efficiently.
# Random sampling helps in initializing the population with diverse solutions.
# Modified spiral dynamics guides the agents towards promising regions of the solution space with adjusted parameters for better performance.
# Swarm intelligence ensures that the population maintains a balance between exploration and exploitation using specific parameters that enhance convergence.