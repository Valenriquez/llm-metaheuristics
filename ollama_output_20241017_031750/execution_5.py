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
# The metaheuristic combines the Gravitational Search (with gravity parameter set to 1.0 and alpha to 0.02) with Random Flight (scaling factor of 1.0, distribution set to levy, and beta of 1.5). The selectors are chosen such that the Gravitational Search uses 'all' selections while the Random Flight employs a probabilistic selection strategy. This combination is used to solve the Rastrigin function optimization problem with two dimensions.
# The Gravitational Search operator leverages gravitational forces to search for better solutions, which can effectively explore the solution space by adjusting the gravity and alpha parameters based on the defined rules.
# The Random Flight operator introduces a random element into the search process by utilizing different distributions (levy in this case). This helps in escaping local minima and exploring new areas of the search space, making it suitable for probabilistic selection.
# Both operators are used with their respective parameter configurations to benefit from the strengths of each approach: gravitational attraction for convergence towards promising regions and random flight for exploration beyond current knowledge.