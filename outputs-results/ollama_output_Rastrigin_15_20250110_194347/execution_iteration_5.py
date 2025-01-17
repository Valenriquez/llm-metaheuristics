# Name: Hybrid Metaheuristic for Rastrigin Function

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.3797444628922573,
            'self_conf': 2.492872690384274,
            'swarm_conf': 2.7210792805218764,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7019872907544645,
            'angle': 13.354514101255447,
            'sigma': 0.02780859453787276
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=31)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines three different operators to explore the search space more effectively. 
# 'random_sample' provides a baseline exploration.
# 'swarm_dynamic' simulates social behavior with updated parameters, which is effective in finding good solutions quickly.
# 'spiral_dynamic' helps refine the search around promising regions with adjusted parameters.
# The combination of these operators ensures a balance between exploration and exploitation, leading to improved performance on the Rastrigin function.