 # Name: GravitationalSearchSwarm
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin2
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
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
# This code defines a metaheuristic named GravitationalSearchSwarm, which uses two operators: gravitational_search and random_flight. 
# The gravitational_search operator is configured with parameters gravity set to 1.0 and alpha set to 0.02. It operates on all possible solutions. 
# The random_flight operator is configured with a scale of 1.0, distribution set to levy, and beta value of 1.5. This operator uses probabilistic selection for its operations. 
# Both operators are chosen based on their respective parameter configurations and the predefined selector settings provided in the parameters_to_take.txt file. 
# The metaheuristic is applied to the Rastrigin2 function, which is a benchmark problem used to evaluate optimization algorithms. The run method is invoked with a specified number of iterations (100), ensuring that the algorithm explores and optimizes the solution space effectively.