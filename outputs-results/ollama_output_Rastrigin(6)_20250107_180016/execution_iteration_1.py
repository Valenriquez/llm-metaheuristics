# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define parameters for each operator
params_random_search = {
    'scale': 0.1,
    'distribution': 'uniform'
}
params_central_force_dynamic = {
    'gravity': 0.01,
    'alpha': 0.5,
    'beta': 2.0
}
params_local_random_walk = {
    'probability': 0.6,
    'scale': 0.8,
    'distribution': 'gaussian'
}

# Define the search operators and their parameters
heur = [
    (
        'random_search',
        params_random_search,
        'greedy'
    ),
    (
        'central_force_dynamic',
        params_central_force_dynamic,
        'probabilistic'
    ),
    (
        'local_random_walk',
        params_local_random_walk,
        'metropolis'
    )
]

# Initialize the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the Random Search, Central Force Dynamic, and Local Random Walk operators.
# The parameters have been chosen to balance exploration and exploitation during the search process.
# By running the metaheuristic multiple times and averaging the results, we can obtain a more robust estimate of the global optimum.