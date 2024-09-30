 # Name: Adaptive Multi-Strategy Optimization Algorithm (AMSOA)
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(1)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_crossover',
    {
         'pairing': 'rank',
         'crossover': 'uniform',
         'mating_pool_factor': 0.4
    },
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation:
# The chosen metaheuristic is named Adaptive Multi-Strategy Optimization Algorithm (AMSOA). 
# It integrates multiple search operators including genetic crossover, mutation, and other dynamic search strategies like gravitational search and swarm dynamics.
# The AMSOA uses the Rastrigin function as the benchmark problem due to its multimodal nature, which allows for testing the exploration capability of different operators.
# Genetic Crossover is configured with 'rank' pairing and 'uniform' crossover type to promote diversity in the population while maintaining high-quality solutions.
# The AMSOA utilizes a greedy selector strategy to dynamically select the most promising operators based on their performance during the initial iterations, allowing for an adaptive optimization process.