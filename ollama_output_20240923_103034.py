## Name: Metaheuristic with Gradient Descent and Crossover

### Code:

# Import all the needed modules (you must also import the benchmark function)
import benchmark_func as bf
import population as pp
import operators as op
import metaheuristic as mh

# Define the benchmark function
fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the metaheuristic
heur = [(
    'genetic_mutation',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 2,
        'factor': 1.0},
    'greedy'  # Selector
), (  # Search operator 2
    'genetic_crossover',  # Perturbator
    {  # Parameters
        'crossover_rate': 0.2,
        'version': 'binomial'
    },
    'probabilistic'  # Selector
)]

# Create the metaheuristic object
met = mh.Metaheuristic(prob, heur, num_iterations=100)

# Run the metaheuristic
met.verbose = True
met.run()

# Print the best solution
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
##### The best selector and operator are greedy and genetic_crossover respectively.

