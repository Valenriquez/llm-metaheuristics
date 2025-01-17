# Name: Hybrid Metaheuristic

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
    (
        'random_search',
        {
            'scale': 0.11535069723900558,
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.009800419481276884,
            'alpha': 0.013956972883611416,
            'beta': 1.1111306163438646,
            'dt': 0.5169452766395723
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8965628276301916,
            'angle': 20.05706952299782,
            'sigma': 0.1342505485286943
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
#met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=20)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines three different search operators: Random Search, Central Force Dynamic, and Spiral Dynamic. Each operator has been configured with specific parameters to suit the problem at hand. The Hybrid Metaheuristic aims to leverage the strengths of each operator by running them concurrently and selecting the best solutions found. The 'all' selector ensures that all agents are considered for selection in each iteration, while the 'probabilistic' and 'greedy' selectors allow for more dynamic changes in the population based on the fitness landscape. This approach helps in exploring different regions of the solution space efficiently and finding better solutions compared to using a single operator.