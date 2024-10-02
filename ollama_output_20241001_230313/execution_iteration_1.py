 # Name: [Your chosen name for the metaheuristic]
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Since the problem is a continuous optimization, we will use a genetic algorithm with crossover and mutation.
# The Rastrigin function is well-suited for global optimization, but it may have multiple local minima which makes it suitable for exploration using both crossover and mutation operators.
heur = [( # Search operator 1
    'add__operator__',  # Perturbator
    {  # Parameters
        'parameters': "genetic_crossover"
    },
    'add__selector__'  # Selector
), ( # Search operator 2
    'add__operator__',  # Perturbator
    {  # Parameters
        'parameters': "genetic_mutation"
    },
    'add__selector__'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Rastrigin function is a continuous optimization problem with multiple local minima, making it suitable for global optimization techniques like genetic algorithms that can explore a wide range of solutions through crossover and mutation operations. 
# We have chosen to use both genetic crossover and genetic mutation as search operators. These operators are appropriate for this type of optimization problem because they allow the algorithm to create new candidate solutions by combining parts of existing ones, which aids in exploring different regions of the solution space where better minima might be found.