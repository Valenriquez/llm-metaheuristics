# Name: Hybrid Adaptive Search Metaheuristic (HASM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.40938938667457253,
            'self_conf': 2.6055056442848787,
            'swarm_conf': 2.5930092587584963,
            'version': 'inertial',
            'distribution': 'levy'
        },
        'metropolis'
    ),
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.2823965489475911,
            'angle': 11.963304774584902,
            'sigma': 0.21017496287424497
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
    met.reset_historicals()
    met.verbose = False
    met.run()

    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Adaptive Search Metaheuristic (HASM) combines the strengths of different search operators to adaptively explore the solution space. It uses a swarm_dynamic operator with inertial version, which is effective for global exploration, random_sample for local refinement, and spiral_dynamic for fine-tuning. This hybrid approach allows for efficient and robust optimization on the Rastrigin function, providing promising results in terms of convergence speed and solution quality.