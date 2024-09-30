 # Name: Metaheuristic Exploration with Custom Operators
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere()  # Example function, change to your target problem
prob = fun.get_formatted_problem()

heur = [(  # Search operator 1
    'genetic_crossover',
    {
        'pairing': 'rank',
        'crossover': 'uniform',
        'mating_pool_factor': 0.4
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Explanation and Justification:
# The code defines a metaheuristic named 'Metaheuristic Exploration with Custom Operators' which utilizes the Sphere function as the benchmark problem. 
# A genetic crossover operator is selected with specific parameters for pairing (rank) and crossover type (uniform). 
# This setup aims to explore various combinations of genetic operators tailored from the provided options, ensuring a diverse search across potential solutions.
# The metaheuristic runs for 100 iterations, with verbose output enabled to track progress and individual operator performance.