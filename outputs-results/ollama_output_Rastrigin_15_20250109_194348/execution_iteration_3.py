# Name: Hybrid Metaheuristic

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
            'scale': 0.01979870720114884,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.009259282364160107,
            'alpha': 0.061935829181420925,
            'beta': 2.1016425037319766,
            'dt': 1.9890702489202914
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7256267652518907,
            'self_conf': 2.6359063863395957,
            'swarm_conf': 2.745207852261716,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()

    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of different search operators to explore the solution space more effectively. The 'random_search' operator helps in exploring new regions, 'central_force_dynamic' encourages convergence towards the optimum by simulating gravitational forces, and 'swarm_dynamic' mimics the behavior of a swarm, allowing for collective exploration and exploitation. By using these operators together with adjusted parameters, the hybrid approach can potentially achieve better results than any single operator alone.