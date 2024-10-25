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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements a gravitational search metaheuristic for optimizing the Rastrigin function with two dimensions. The Gravitational Search algorithm is designed to mimic the gravitational interactions between masses, where each 'mass' represents a potential solution in the search space. The parameters include gravity (strength of the gravitational force) and alpha (a scaling factor). The selector 'all' means that this operator will be applied to all candidate solutions during each iteration. This approach aims to explore the search space more thoroughly by simulating the forces between masses, promoting convergence towards better fitness regions.
