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
        'metropolis'
    ),
    (  
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is named GravitationalSearchMetaheuristic as it incorporates the gravitational search operator, which is a type of optimization algorithm inspired by the principles of physics, specifically gravity. The algorithm uses two main operators: "gravitational_search" and "random_flight". 

# The gravitational_search operator has parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. These parameters control the strength of the gravitational force and the rate at which the population adjusts its positions, respectively. The selector for this operator is set to 'metropolis', indicating that it will be used in a probabilistic manner during the search process.

# The random_flight operator is designed with parameters 'scale' set to 1.0, 'distribution' set to 'levy', and 'beta' set to 1.5. These settings influence how the population explores the solution space; the scale determines the range of exploration, while 'distribution' defines whether this exploration follows a levy or other distribution (in this case, a levy distribution). The selector for random_flight is 'probabilistic', ensuring that its actions are guided by probabilistic rules during optimization.

# These two operators together provide a balance between systematic search and random exploration, which should help in efficiently finding the optimal solution to the Rastrigin function within the given number of iterations (100). The use of both gravitational search and random flight ensures a comprehensive approach to solving complex optimization problems like the one represented by the Rastrigin function.