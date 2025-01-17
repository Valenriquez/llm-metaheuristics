# Name: Hybrid Metaheuristic for Global Optimization
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
    ('random_search',
        {
            'scale': 0.3780700568903595,
            'distribution': 'uniform'
        },
        'greedy'),
    ('swarm_dynamic',
        {
            'factor': 0.5208726602361291,
            'self_conf': 2.6819722798367507,
            'swarm_conf': 2.4226051313133,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'),
    ('spiral_dynamic',
        {
            'radius': 0.5030286417166663,
            'angle': 29.599443827298664,
            'sigma': 0.032257964884433214
        },
        'greedy')
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of three different optimization algorithms: 
# - Random Search provides a broad exploration of the search space, helping to avoid local optima.
# - Swarm Dynamic mimics the behavior of social animals, allowing for both exploitation and exploration.
# - Spiral Dynamic uses a spiral path to efficiently search around promising areas.
# The combination aims to balance exploration and exploitation, potentially leading to better solutions than using any single algorithm alone.