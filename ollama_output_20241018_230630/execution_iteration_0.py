 # Name: GravitationalSearchAndRandomFlight
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
        'greedy'
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
# The metaheuristic GravitationalSearchAndRandomFlight is designed to leverage both gravitational search and random flight operators for optimization. 
# Gravitational Search, with its gravity parameter set to 1.0 and alpha to 0.02, aims to simulate the gravitational force acting on particles in a multi-dimensional space, promoting convergence towards better solutions.
# Random Flight, characterized by scale of 1.0, distribution set to 'levy', and beta value of 1.5, is used to introduce diversity into the search process by exploring new regions not covered by the gravitational force alone. The levy distribution helps in navigating rugged landscapes effectively.
# Both operators are configured with specific parameters as per the requirements from parameters_to_take.txt, ensuring that the metaheuristic operates within defined constraints and objectives.
# The 'greedy' selector for gravitational search ensures that only the most promising solutions are retained, while the 'probabilistic' selector for random flight allows for a controlled exploration of diverse solution spaces, combining deterministic and stochastic elements for better convergence properties.