# Name: HybridMetaheuristic

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
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'probabilistic'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 2,
            'scaling_factor': 0.8
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 30,
            'sigma': 0.1
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1500, num_agents=205)
# met.verbose = True # please comment this line
# met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1500, num_agents=205)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HybridMetaheuristic combines several search operators to enhance the exploration and exploitation capabilities of the solution space. It starts with a random search operator to explore the space, followed by a central force dynamic operator to exploit promising regions, a differential mutation operator to introduce diversity, and a spiral dynamic operator for continuous exploration. The combination of these operators helps in finding optimal solutions more efficiently for the Rastrigin function.