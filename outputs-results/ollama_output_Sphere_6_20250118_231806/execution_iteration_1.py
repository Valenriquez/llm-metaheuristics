# Name: Randomized Adaptive Metaheuristic (RAM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(6) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.014378508489900542,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6640954703691787,
            'self_conf': 2.225307429677205,
            'swarm_conf': 2.913545910789139,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6122896532767045,
            'angle': 22.499740268707306,
            'sigma': 0.12555299920916652
        },
        'greedy'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Randomized Adaptive Metaheuristic (RAM) combines several search operators to explore the solution space more effectively. 
# It starts with a local search approach using 'random_search' to find good initial solutions, utilizing a Gaussian distribution for better exploitation around promising areas.
# Then, it integrates a particle swarm optimization ('swarm_dynamic') guided by a constriction version to balance exploration and exploitation efficiently.
# Finally, 'spiral_dynamic' is added to exploit promising regions around the current best solutions, enhancing the search process. 
# This combination aims to balance exploration and exploitation, potentially leading to better convergence and robustness in finding the global optimum.