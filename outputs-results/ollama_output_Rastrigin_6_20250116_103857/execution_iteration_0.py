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
            'scale': 0.1399836979723491,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.736222954640493,
            'self_conf': 1.9934384528011502,
            'swarm_conf': 1.9712876676563744,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5900410154721194,
            'angle': 12.70828955618993,
            'sigma': 0.17767712137658212
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)

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
# The HybridMetaheuristic combines three search operators: Random Search, Swarm Dynamic, and Spiral Dynamic.
# Random Search helps to explore the solution space without being trapped in local optima.
# Swarm Dynamic mimics the behavior of a swarm of particles moving together in search of food, adapting based on individual experience (self_conf) and group experience (swarm_conf).
# Spiral Dynamic uses a spiral movement strategy, which can efficiently search for solutions by expanding outward from the center.
# The combination of these operators allows for a robust exploration and exploitation balance, potentially leading to better optimization results.