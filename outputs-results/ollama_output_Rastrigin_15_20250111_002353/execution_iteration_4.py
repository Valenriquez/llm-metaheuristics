# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'greedy'),
    ('local_random_walk', {'scale': 0.43711166501174775, 'probability': 0.6446136180337296}, 'metropolis'),
    ('swarm_dynamic', {'factor': 0.7190768719795838, 'self_conf': 2.2874278410649844, 'swarm_conf': 2.565665426351245}, 'probabilistic'),
    ('spiral_dynamic', {'radius': 0.7489733486040195, 'angle': 20.937789561021145, 'sigma': 0.10278784085020343}, 'all'),
]

met = mh.Metaheuristic(prob, heur, num_iterations=2000, num_agents=300)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 5 times
for rep in range(5):
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HybridMetaheuristic combines different search operators such as random sampling, local random walk, swarm dynamics, and spiral dynamic. 
# The selection of operators is driven by their potential to explore and exploit the solution space effectively.
# Each operator has been tailored with specific parameters to balance exploration and exploitation.
# The metropolis selector is used for operators that can produce solutions with higher fitness but risk of getting stuck in local optima.
# Running multiple iterations and agents allows for a more thorough exploration of the solution space, reducing the chances of premature convergence.