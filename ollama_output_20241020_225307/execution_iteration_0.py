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
# This metaheuristic combines the Gravitational Search Algorithm (GSA) with a Random Flight mechanism to explore potential solutions in the search space. 
# The GSA uses gravity as a parameter to attract particles towards better regions of the solution space, while random flight introduces stochastic elements by allowing for probabilistic selection based on the distribution type specified.
# The Gravitational Search component is applied to all candidates ('all' selector), whereas the Random Flight operator utilizes a probabilistic approach ('probabilistic' selector) which selects solutions with a probability proportional to their fitness. 
# This hybridization aims to leverage both deterministic gravitational attraction and stochastic random flight for effective exploration of the solution space, potentially leading to better convergence towards an optimal solution than either method alone.