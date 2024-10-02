 Now create the metaheuristic:
# Name: Custom Genetic Algorithm with Rastrigin Function
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    (
        'GeneticAlgorithm',
        {
            'population_size': 100,
            'mutation_probability': 0.05,
            'crossover_rate': 0.8,
            'max_iterations': 100,
            'elitism_count': 2,
        },
        'Standard'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a custom genetic algorithm to solve the Rastrigin function with 2 dimensions.
# The population size is set to 100, indicating that 100 candidate solutions are generated initially.
# The mutation probability is set to 0.05, meaning each gene has a 5% chance of being mutated during reproduction.
# The crossover rate is 0.8, which means that on average, 80% of the genes from two parents will be combined in the offspring.
# Elitism count is set to 2, ensuring that the top 2 individuals are directly carried over to the next generation without any changes.
# These settings aim to balance exploration (mutation) and exploitation (crossover) in the genetic algorithm process.