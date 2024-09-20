# Name: Metaheuristic with Greedy and Differential Mutation

# Code:
import benchmark_func as bf
import population as pp
import operators as op
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [(  # Search operator 1
    'differential_mutation',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 2,
        'factor': 1.0},
    'greedy'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Text:
# This metaheuristic uses the greedy selector with the differential mutation operator.
# It has 100 iterations and is designed to find the optimal solution for the Rastrigin function.