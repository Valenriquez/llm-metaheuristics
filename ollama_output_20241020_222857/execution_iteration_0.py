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
# This code implements a Gravitational Search metaheuristic for optimizing the Rastrigin function with two dimensions. 
# The Gravitational Search algorithm is used with parameters gravity set to 1.0 and alpha to 0.02. 
# The selector 'all' means that all possible candidates will be considered during the search, not just a subset as in other selectors like greedy or probabilistic. 
# This setup aims to explore the solution space more thoroughly by considering multiple potential solutions at each iteration.
# The Rastrigin function is chosen because it is commonly used for testing optimization algorithms due to its many local minima, which allows for comprehensive evaluation of search strategies.