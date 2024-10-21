 # Name: GravitationalSearchMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# This code implements a Gravitational Search Metaheuristic using the Rastrigin function with 2 dimensions. The gravitational_search operator is used, with parameters gravity set to 1.0 and alpha to 0.02. The selector 'all' means that this operator will be applied to all elements in the population during each iteration. This choice is based on the parameter values from parameters_to_take.txt, which specifies a standard configuration for gravitational search algorithms.
