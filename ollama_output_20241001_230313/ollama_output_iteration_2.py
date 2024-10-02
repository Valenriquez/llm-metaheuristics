 # Name: Custom Metaheuristic with Genetic Algorithm
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'add__operator__',
    {
        'parameters': ""
    },
    'add__selector__'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a custom metaheuristic using the Rastrigin function as the benchmark problem. 
# We use a genetic algorithm with default parameters for both crossover and mutation operators. 
# Since we are implementing a genetic algorithm, it is mandatory to include both genetic crossover and genetic mutation. 
# The selector used here is 'add__selector__', which implies that the search operator being applied is based on adding elements as per the selector's definition from parameters_to_take.txt.