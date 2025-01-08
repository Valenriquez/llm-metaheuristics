# Name: Multi-Objective Metaheuristic Algorithm (MOMA)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.05,
            'beta': 1.5
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This metaheuristic combines four different search operators: Random Search, Central Force Dynamic, Swarm Dynamic, and Spiral Dynamic. Each operator has been configured with specific parameters to enhance its exploration and exploitation capabilities. The selection of 'metropolis' as the selector for Random Search allows it to escape local minima effectively. The probabilistic selector for Central Force Dynamic encourages a balance between exploration and exploitation by randomly choosing new positions based on the calculated force. The greedy selector for Swarm Dynamic ensures that each agent follows its best known path, which can lead to rapid convergence in certain regions of the search space. The Spiral Dynamic operator is included to introduce additional diversity and help escape local optima.

# The algorithm runs for 100 iterations and 30 repetitions to ensure robustness and reliability. The final fitness values are collected and analyzed to provide insights into the performance of the metaheuristic on the Rastrigin function.