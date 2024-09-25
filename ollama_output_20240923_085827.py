# Name: Hybrid Metaheuristic
# Code:

import benchmark_func as bf
import population as pp
import operators as op
import metaheuristic as mh


fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operators
heur = [(
    'differential_mutation',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 2,
        'factor': 1.0},
    'greedy'  # Selector
), (  # Search operator 2
    'differential_crossover',  # Perturbator
    {  # Parameters
        'crossover_rate': 0.2,
        'version': 'binomial'
    },
    'greedy'  # Selector
)]

# Create the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Run the metaheuristic
met.run()

# Print the best solution
print('x_best = {}, f_best = {}'.format(*met.get_solution()))


# This metaheuristic combines the differential mutation and differential crossover operators, 
# using the greedy selector. The greedy selector chooses the best solution from the population at each iteration.
# This hybrid metaheuristic is expected to perform better than the individual operators, 
# as it combines their strengths and overcomes their weaknesses.