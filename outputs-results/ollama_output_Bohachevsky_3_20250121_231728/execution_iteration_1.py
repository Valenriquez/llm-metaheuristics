# Name: Hybrid Metaheuristic for Bohachevsky(3)

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(3) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.03246424844821945,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0013370429429817021,
            'alpha': 0.012170440628050642,
            'beta': 1.5108126405835436,
            'dt': 1.0041079581946954
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)  
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The hybrid metaheuristic combines a random search operator with a central force dynamic operator. 
# The random search helps to explore the solution space, while the central force dynamic encourages convergence towards optimal solutions.
# By using both operators in parallel, the hybrid approach aims to balance exploration and exploitation effectively.
# The metropolis selector allows for stochastic acceptance of new solutions, further enhancing the exploration capabilities of the algorithm.