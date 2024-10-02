 # Name: Custom Genetic Algorithm with Swarm Dynamic Optimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operators with parameters from parameters_to_take.txt
heur = [( # Search operator 1: Genetic Crossover
    'genetic_crossover',
    {
        'rate': [0.9],
        'distribution': ['uniform'],
    },
    'greedy'
), ( # Search operator 2: Genetic Mutation
    'genetic_mutation',
    {
        'rate': [0.1],
        'angle': [22.5],
        'sigma': [0.1],
    },
    'metropolis'
)]

# Define the metaheuristic with the operators and parameters specified
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The algorithm is designed as a custom genetic algorithm that combines both genetic crossover and mutation to explore the solution space effectively. 
# Genetic Crossover uses a uniform distribution with a rate of 0.9, promoting diversity in the population by exchanging segments between chromosomes. 
# Genetic Mutation employs an angle of 22.5 degrees and a sigma value of 0.1 for mutation, which allows fine-tuning individual traits within the population. 
# Both operators are selected using the greedy selector to prioritize exploration over exploitation in each iteration.
# This combination aims to balance between global exploration and local refinement, ensuring that the algorithm can navigate complex landscapes efficiently.