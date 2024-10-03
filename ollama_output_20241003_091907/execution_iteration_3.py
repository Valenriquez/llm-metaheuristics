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
            'probability': 0.8,
            'mutation_rate': 0.1,
        },
        'greedy'
    ),
    ( # Search operator 2
        'genetic_mutation',
        {
            'probability': 0.2,
            'mutation_rate': 0.3,
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Custom Genetic Algorithm leverages both genetic crossover and mutation for exploration. 
# Crossover combines parts of two parent solutions to create offspring, promoting diversity in the population.
# Mutation introduces random changes to individual solutions, ensuring that local optima are avoided and allowing for broader search.
# The 'greedy' selector focuses on improving the fittest individuals, while 'all' considers all candidates for selection.
# These settings aim to balance exploration (crossover) with exploitation (mutation and greedy selection), suitable for continuous optimization problems like Sphere function evaluation.