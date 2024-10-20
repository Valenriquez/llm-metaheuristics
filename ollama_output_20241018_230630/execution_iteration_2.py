 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(22)
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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm. The algorithm is designed to solve optimization problems by simulating the effect of gravity on particles within a system, with parameters set as specified in the parameters_to_take.txt file. The selector 'all' indicates that this operator should be applied to all potential solutions during the search process. The Gravitational Search Operator uses two main parameters: 'gravity' and 'alpha'. These parameters control the strength of the gravitational force and its influence on the search space, respectively.
# By setting the gravity parameter high, we encourage exploration of a wider area in the search space, while the alpha parameter helps to fine-tune this exploration by adjusting the step size during each iteration. This approach aims to balance between global and local explorations to efficiently find optimal solutions for the given benchmark function.