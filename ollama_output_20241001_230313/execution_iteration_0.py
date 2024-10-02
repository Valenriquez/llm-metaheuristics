 # Name: Custom Genetic Algorithm with Multiple Operators
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_mutation',
    {
        'scale': 0.5,
        'elite_rate': 0.1,
        'mutation_rate': 0.25,
        'distribution': 'uniform'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and justification:
# The metaheuristic is designed to solve the Rastrigin function optimization problem using a custom Genetic Algorithm with multiple operators. 
# A genetic algorithm is chosen because it mimics biological evolution, allowing for exploration of multiple solutions through selection, crossover, and mutation.
# The primary operator used here is 'genetic_mutation', which introduces variation in the population by adjusting the scale, elite rate, mutation rate, and distribution method as specified. 
# These parameters are selected from the provided options to ensure a balance between exploration (high diversity) and exploitation (focused search around current best solutions).
# The selector 'probabilistic' is used because it allows for random selection of individuals within the population according to predefined probabilities, which promotes diversity in mutations while considering the fitness of each individual. 
# This setup helps in navigating the multi-modal function landscape characteristic of the Rastrigin function by encouraging exploration and exploitation across iterations.