# Name: Enhanced Metaheuristic

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
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.011690771955423849,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0009619944448479293,
            'alpha': 1.414196720998301,  
            'beta': 2.627812990943991   
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 1.00435940998965,
            'self_conf': 2.399163401270318,
            'swarm_conf': 2.3164441461224703,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines three different search operators: random search, central force dynamic, and swarm dynamic. The choice of these operators and their parameters is based on their known effectiveness in handling various types of optimization problems. The use of 'greedy' selector for random search helps in quickly converging to a feasible solution, while the 'all' selector for central force dynamic allows for exploration across the entire search space. The 'metropolis' selector for swarm dynamic ensures that the algorithm can escape local minima effectively. Running multiple iterations and averaging the final fitness results provides a more robust assessment of the metaheuristic's performance on the given problem.