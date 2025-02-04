# Name: Hybrid Metaheuristic with Differential Mutation and Firefly Dynamic
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (
        'differential_evolution',
        {
            'F': 0.8,
            'CR': 0.9,
            'num_candidates': 30
        },
        'random_sample'
    ),
    (
        'firefly_algorithm',
        {
            'alpha': 1.5,
            'beta0': 1.0,
            'gamma': 2.0,
            'iteration_coefficient': 0.98,
            'n_iterations_without_improvement': 100
        },
        'random_sample'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the differential evolution (DE) and firefly algorithm (FA).
# DE is used for global search with its mutation and crossover operations.
# FA is incorporated to improve local search by mimicking the flashing behavior of fireflies.
# Both operators are selected with a random sample selector, allowing them to explore and exploit the solution space effectively.