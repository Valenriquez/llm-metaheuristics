 # Name: Custom Genetic Algorithm
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'genetic_crossover',
    { 
        'probability': 0.9,
        'elitism': True,
        'mutation_rate': 0.05
    },
    'greedy'
    ),
    ( # Search operator 2
    'genetic_mutation',
    { 
        'probability': 0.1,
        'mutation_rate': 0.05
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named "Custom Genetic Algorithm" uses a genetic crossover and mutation approach to explore the solution space. 
# The genetic_crossover operator is configured with a high probability of 0.9 for recombination, enabled elitism which ensures that the best individuals are preserved in the population, 
# and a mutation rate of 0.05 to introduce diversity. The genetic_mutation operator has a lower probability of 0.1 for individual gene changes within chromosomes, also with a mutation rate of 0.05.
# These operators work in conjunction with the 'greedy' and 'metropolis' selectors respectively, which guide the search towards better solutions based on predefined criteria.
# The selection of these parameters and operators is informed by typical settings used in genetic algorithms that balance exploration and exploitation to efficiently solve optimization problems like the Sphere function.