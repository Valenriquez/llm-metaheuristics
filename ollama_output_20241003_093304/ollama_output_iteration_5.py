 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
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
# This metaheuristic is designed based on the Gravitational Search algorithm, which mimics the gravitational force among masses to search for optimal solutions. The parameters used are 'gravity' set to 1.0 and 'alpha' set to 0.02, as specified in the parameters_to_take.txt file. The selector 'all' is chosen because we want this algorithm to apply all operators at each step during the search process. This approach allows for a balance between exploration (random search) and exploitation (local optimization), which is typical of gravitational systems where heavier masses attract lighter ones, mimicking an optimal solution seeking behavior in our problem space. The use of these parameters aims to leverage both local and global search capabilities, potentially leading to better convergence towards the global minimum as defined by the benchmark function Sphere.