# Name: Enhanced Differential Mutation with Probabilistic Selector
# Code:

# Import all the needed modules
import benchmark_func as bf
import population as pp
import operators as op
import metaheuristic as mh

# Define the function to optimize
fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the metaheuristic
heur = [(
    'differential_mutation',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 2,
        'factor': 1.0
    },
    'probabilistic'  # Selector
), (
    'differential_crossover',  # Perturbator
    {  # Parameters
        'crossover_rate': 0.2,
        'version': 'binomial'
    },
    'probabilistic'  # Selector
)]

# Create the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Run the metaheuristic
met.run()

# Print the best solution
print('x_best = {}, f_best = {}'.format(*met.get_solution()))


# Text after the code:

# The Enhanced Differential Mutation with Probabilistic Selector algorithm uses the differential mutation operator to generate new solutions and the probabilistic selector to choose the best solutions.
# This metaheuristic is expected to perform well for optimization problems with a high dimensionality.