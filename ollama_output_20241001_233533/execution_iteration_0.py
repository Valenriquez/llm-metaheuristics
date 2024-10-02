 # Name: Adaptive Genetic Algorithm with Random Mutation and Crossover
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Using a combination of genetic mutation and crossover for exploration and exploitation
# Genetic Mutation with Gaussian Distribution and Metropolis Selection for Exploration
# Crossover can be introduced to create new solutions, enhancing the diversity in the population
heur = [
    ('genetic_mutation', {
        'parameters': {
            'scale': 0.5,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        }}, 'metropolis')
]

# Adding a simple crossover operator for diversity and potential improvement in the population
heur += [('random_sample', {}, 'all')] # Random Crossover to mix genetic material from different individuals

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is an Adaptive Genetic Algorithm with a focus on using Gaussian mutation for exploration and random crossover to enhance diversity. The Metropolis selector ensures that only high-quality solutions are selected, promoting convergence towards better regions of the search space. By including both genetic mutation and random_sample (crossover) operators, we aim to balance between exploring new areas and exploiting known good solutions. This approach allows for a more adaptive search process where each step can adjust based on the population's performance, improving overall efficiency in optimization tasks like minimizing the Rastrigin function.
