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
    ),
    ( # Local Random Walk Operator
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using three main operators from the parameters_to_take.txt file. 
# The gravitational search operator is designed with gravity set to 1.0 and alpha to 0.02, while the random flight uses scale of 1.0 with a levy distribution and beta of 1.5. 
# The local random walk employs a probability of 0.75, scale of 1.0, and a uniform distribution. All operators use probabilistic selection as specified in parameters_to_take.txt to guide the search process. 
# These operators are applied iteratively for 100 iterations to solve the Rastrigin function optimization problem, with verbose output enabled for monitoring convergence.
