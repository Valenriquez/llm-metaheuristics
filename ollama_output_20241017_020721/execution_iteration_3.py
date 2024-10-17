 # Name: GravitationalSearchAndRandomFlightMetaheuristic
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
# This metaheuristic combines the Gravitational Search algorithm with the Random Flight operator. 
# The Gravitational Search, characterized by its gravity parameter (1.0) and alpha value (0.02), is used to explore the search space effectively. 
# It aims to simulate the gravitational force among particles in a multidimensional space, allowing for convergence towards better solutions.
# The Random Flight operator, with scale set to 1.0, employs a distribution of 'levy' and beta value (1.5) is used to control the probabilistic selection of movements. This helps in avoiding local minima by introducing random changes in the search trajectory, promoting global exploration.
# Both operators are applied using the 'all' selector for continuous application throughout the iterations and 'probabilistic' selector for the Random Flight operator to influence its behavior with a probability-based approach. 
# This combination is expected to balance between exploitation of promising areas and exploration of new spaces, making it suitable for optimization tasks where both aspects are crucial.