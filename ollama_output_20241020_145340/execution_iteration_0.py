 # Name: GravitationalSearchAndRandomFlightMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin
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
# The Gravitational Search And Random Flight Metaheuristic is designed to leverage the strengths of both gravitational search and random flight operators. 
# Gravitational search, characterized by its 'gravity' parameter set to 1.0 and 'alpha' set to 0.02, is used to explore potential solutions in a manner that mimics the behavior of celestial bodies under gravitational influence. This helps in converging towards promising regions of the search space.
# Random flight, with parameters including 'scale' set to 1.0, 'distribution' as 'levy', and 'beta' set to 1.5, introduces stochasticity by allowing solutions to explore diverse areas of the search space. The 'levy' distribution ensures that this exploration is both random and directed, aiding in escaping local minima effectively.
# Both operators are utilized with a common selector 'probabilistic'. This choice allows for a balance between deterministic (gravitational search) and stochastic (random flight) components, enhancing the algorithm's ability to navigate complex landscapes while avoiding premature convergence. 
# The combination of gravitational attraction and random explorations enables this metaheuristic to effectively optimize the Rastrigin function by balancing exploration with exploitation.