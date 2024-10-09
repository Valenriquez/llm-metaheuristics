 # Name: GravitationalSearchAndRandomFlightMetaheuristic
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
# The Gravitational Search And Random Flight Metaheuristic is designed to leverage both gravitational search and random flight operators for optimization. 
# Gravitational search, with its gravity parameter set to 1.0 and alpha to 0.02, influences the exploration of solution space by simulating the gravitational force between masses.
# Random flight, characterized by a scale factor of 1.0, employs either levy or uniform distribution for random movements, which aids in the diversification of search. The beta parameter in random flight is set to 1.5, enhancing its ability to escape local minima.
# Both operators are used throughout the iterations with a selector setting 'all' and 'probabilistic', respectively. This approach aims to balance between exploration and exploitation, ensuring comprehensive coverage of the solution space while maintaining a focus on potentially optimal regions through probabilistic selection.