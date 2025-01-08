# Name: Adaptive Multi-Operator Metaheuristic (AMOM)
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
            'scale': 0.1,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.002,
            'alpha': 0.02,
            'beta': 1.6
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8,
            'self_conf': 3.04,
            'swarm_conf': 3.06,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 25.0,
            'sigma': 0.12
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=200, num_agents=150)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic employs a diverse set of search operators to explore the solution space efficiently. The 'random_search' operator helps in escaping local minima, while 'central_force_dynamic', 'swarm_dynamic', and 'spiral_dynamic' provide more sophisticated exploration strategies. By combining these operators with different selectors, AMOM is designed to adaptively balance exploration and exploitation, making it suitable for complex optimization problems like the Rastrigin function.