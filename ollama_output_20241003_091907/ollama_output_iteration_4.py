 # Name: Adaptive Random Search with Genetic Operators
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
            'elitism': True,
            'mutation_rate': 0.1
        },
        'greedy'
    ),
    ( # Search operator 2
        'genetic_mutation',
        {
            'probability': 0.3,
            'mutation_rate': 0.2
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is named "Adaptive Random Search with Genetic Operators." This approach combines genetic crossover and mutation, which are based on the probability of exchange and mutation within a population. Crossover promotes diversity by exchanging parts of two parent solutions, while mutation introduces random changes to maintain exploration in areas of low density. 
# The 'greedy' selector is used for crossover to favor immediate improvements, enhancing convergence speed. For mutation, the 'metropolis' selector based on acceptance probability helps balance exploration and exploitation, preventing premature convergence. Parameters like probability and rates are set according to typical values recommended for these genetic operators in literature, ensuring a balance between exploration and exploitation as required by the task description.