## Name: Metaheuristic using Genetic Mutation with Greedy Selection

## Code:
import sys


sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import population as pp
import operators as op
import metaheuristic as mh


fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [(
    'genetic_mutation',  # Perturbator
    {  # Parameters
        'scale': 1.0,
        'elite_rate': 0.1,
        'mutation_rate': 0.25,
        'distribution': 'gaussian',
    },
    'greedy'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
