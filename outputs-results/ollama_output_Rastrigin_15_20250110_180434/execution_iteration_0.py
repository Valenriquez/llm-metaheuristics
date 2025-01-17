# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.018490815491456557,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.006177654912975475,
            'alpha': 0.02995664071381697,
            'beta': 1.5370788327384435
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8470528817405123,
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=500)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1500, num_agents=49)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HybridMetaheuristic combines multiple search operators to improve exploration and exploitation capabilities. The random_search operator helps in exploring the solution space, central_force_dynamic guides particles towards optimal solutions with specific gravity, alpha, and beta parameters, and local_random_walk aids in fine-tuning around the best solutions found so far. This combination is expected to lead to better convergence and a more robust search process for the Rastrigin function.