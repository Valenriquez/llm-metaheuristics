# Name: Hybrid Metaheuristic using Random Search and Firefly Dynamic

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
            'scale': 0.01,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 5.0
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=500, num_agents=30)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines Random Search and Firefly Dynamic to explore the search space more effectively. 
# The Random Search operator is used for initial exploration with a Gaussian distribution and small scale, while the Firefly Dynamic operator is applied for local refinement with a uniform distribution and parameters that encourage convergence.
# Running multiple iterations and replicates helps in better understanding the performance and robustness of the metaheuristic on the Rastrigin function.