# Name: Multi-Operator Hybrid Metaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.06858479916409348,
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7984545097913769,
            'self_conf': 2.69905573237369,
            'swarm_conf': 2.018361617721047,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.704609524196219,
            'angle': 24.953411675998133,
            'sigma': 0.09772408454620213
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This multi-operator hybrid metaheuristic combines the strengths of different search operators to explore the solution space more effectively. The `random_search` operator helps in exploring new areas with a Gaussian distribution, while the `swarm_dynamic` operator leverages social behavior for faster convergence using constriction version. The `spiral_dynamic` operator facilitates a systematic exploration towards the optimal solution. By using a combination of these operators and the Metropolis selector, the algorithm can efficiently balance exploration and exploitation, leading to better performance on the Rastrigin function.