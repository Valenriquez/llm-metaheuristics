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
# This metaheuristic combines the gravitational search algorithm with a random flight strategy to explore the solution space. The Gravitational Search operator uses parameters 'gravity' (1.0) and 'alpha' (0.02) to control the attraction between particles, while the Random Flight operator uses 'scale' (1.0), 'distribution' ('levy'), and 'beta' (1.5). Both operators are applied using the 'probabilistic' selector to balance exploration and exploitation in the search process. The combination is designed to leverage both global and local search capabilities, potentially leading to better convergence towards an optimal solution for the Rastrigin function.
