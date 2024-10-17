 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with two main operators: gravitational search and random flight. 
# The gravitational search operator uses parameters 'gravity' set to 1.0 and 'alpha' to 0.02, while the selector for this operator is set to 'metropolis'. 
# The random flight operator employs 'scale' of 1.0, with a distribution type of 'levy' and a beta value of 1.5. This operator uses 'probabilistic' as its selector. 
# These parameters are chosen based on the settings from the parameters_to_take.txt file to ensure optimal performance for the Rastrigin function optimization task. The use of both operators with their respective parameterizations and selectors is justified by the need to explore diverse search spaces and leverage different probabilistic mechanisms in the optimization process.