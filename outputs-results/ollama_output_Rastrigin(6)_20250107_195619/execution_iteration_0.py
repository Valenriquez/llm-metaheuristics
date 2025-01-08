# Name: Rastrigin Search with Hybrid Operators

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
    ('random_sample', {}, 'greedy'),
    ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'metropolis'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 'probabilistic')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True
# met.run()

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The metaheuristic is designed to explore the Rastrigin function using a combination of different search operators. 
# `random_sample` is used as a baseline operator with default parameters.
# `spiral_dynamic` is included to guide the population towards better regions by simulating spiral motion.
# `swarm_dynamic` incorporates swarm intelligence, mimicking the behavior of particles in a dynamic environment. This hybrid approach aims to balance exploration and exploitation effectively.