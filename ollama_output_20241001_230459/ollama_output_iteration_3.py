 # Name: Custom Genetic Algorithm
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Search operator 1: Genetic Algorithm with Inertial Weighted Evolution Strategy (GSIWES)
# Parameters based on the provided JSON structure, ensuring all parameters are from parameters_to_take.txt
heur = [(
    'genetic_crossover',
    {
        'mutation_rate': 0.1,
        'population_size': 50,
        'elitism_count': 2,
        'generations': 100
    },
    'metropolis'
)]

# Search operator 2: Random Search Algorithm
heur += [(
    'random_search',
    {
        'sample_size': 50,
        'max_attempts': 200
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is a Custom Genetic Algorithm designed to solve the Rastrigin function optimization problem with two main operators: 
# 1. Genetic Crossover, which incorporates mutation based on a predefined rate (0.1) and maintains a population size of 50 individuals while preserving top 2 from each generation through elitism. This approach is iterated over 100 generations to refine the solution. The selector 'metropolis' ensures that only promising solutions are carried forward, mimicking a probabilistic selection process in high-dimensional spaces.
# 2. Random Search Algorithm, which explores potential solutions by sampling the search space randomly (50 samples) up to 200 attempts. This approach is characterized by its probabilistic nature and contributes to an overall exploration of diverse solution landscapes during optimization.
# These two operators together provide a balanced approach between exploitation of current promising areas and exploration of new regions in the search space, ensuring robust optimization across both local and global optima for the Rastrigin function.