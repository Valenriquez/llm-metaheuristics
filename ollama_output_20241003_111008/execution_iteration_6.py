 # Name: GravitationalSearchAndRandomFlight
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
    ( # Random Flight Operator
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
# The GravitationalSearchAndRandomFlight metaheuristic is designed to leverage both gravitational search and random flight strategies for optimization. 
# The Gravitational Search Operator, with a gravity parameter set to 1.0 and an alpha value of 0.02, guides the exploration towards better solutions by simulating the effect of gravity on masses. 
# This operator uses all possible solutions in its operation.
# On the other hand, the Random Flight Operator employs a scale factor of 1.0, with the distribution set to 'levy', and operates probabilistically. 
# The levy distribution adds diversity to the search by allowing for random jumps that are more likely to escape local minima.
# Both operators contribute to the exploration and exploitation balance required in metaheuristic optimization, providing a comprehensive approach to finding optimal solutions for the given benchmark function.