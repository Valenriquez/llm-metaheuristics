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
# This metaheuristic combines the gravitational search algorithm with a random flight strategy to explore the solution space. The Gravitational Search Operator uses parameters gravity (1.0) and alpha (0.02) to control the strength of the gravitational force and the influence of local minima, respectively. It applies this operator to all possible solutions ('all'). The Random Flight Operator introduces stochastic elements by scaling (1.0), selecting a distribution type ('levy') for exploration, and setting beta (1.5). This is applied selectively using the 'probabilistic' selector, which means it will only be used with probabilistic methods to ensure diversity in the search process. Together, these operators aim to balance between exploitation of promising areas and exploration of new regions in the solution space, leveraging both deterministic and stochastic components for effective optimization.