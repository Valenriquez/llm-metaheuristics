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
# The metaheuristic named GravitationalSearchAndRandomFlightMetaheuristic is designed to solve optimization problems using a combination of gravitational search and random flight operators. 
# Gravitational search simulates the behavior of celestial bodies under the influence of gravity, which helps in exploring the solution space by attracting particles towards promising regions. 
# Random flight introduces a probabilistic element that allows for exploration beyond local minima by utilizing operators like levy distribution or uniform distribution based on beta parameter.
# This combination aims to leverage both the exploitative power of gravitational search and the explorative capability of random flight, providing a robust approach to finding optimal solutions in complex landscapes.