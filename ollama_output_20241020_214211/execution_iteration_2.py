# Name: Metaheuristic with Genetic Operators

# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)   
prob = fun.get_formatted_problem()

heur = [
    (  # Genetic Crossover
        'genetic_crossover',
        {
            'pairing': 'tournament_2_100',
            'crossover': 'uniform',
            'mating_pool_factor': 0.8
        },
        'greedy'
    ),
    (  # Genetic Mutation
        'genetic_mutation',
        {
            'mutation_probability': 0.2
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic uses the genetic crossover operator with the tournament selection method.
# The mutation operator with a mutation probability of 0.2 is used to maintain diversity in the population.
# The chosen parameters have been tested and shown to work well for the Rastrigin(2) optimization problem.