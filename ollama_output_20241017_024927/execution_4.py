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
# This code implements a Gravitational Search metaheuristic for solving the Rastrigin function with two dimensions. The Gravitational Search algorithm is used to explore the search space, leveraging gravitational forces between particles to guide their movement towards better solutions. Parameters include 'gravity' set to 1.0 and 'alpha' set to 0.02, which control the strength of the gravitational force and influence the exploration vs exploitation trade-off, respectively. The selector is set to 'all', meaning all operators will be applied in each iteration. This choice aims to balance between global search (exploration) and local search (exploitation), enhancing the algorithm's ability to converge towards the optimal solution while avoiding premature convergence.
