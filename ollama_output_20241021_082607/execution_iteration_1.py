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
# The gravitational search uses a gravity parameter of 1.0 and an alpha value of 0.02, affecting its exploration capabilities.
# The random flight operator is scaled to 1.0 and utilizes a levy distribution for diversification, especially beneficial in global optimization scenarios.
# Both operators are set to operate on all possible solutions ('all'), with the gravitational search using a greedy selector by default, while the random flight employs a probabilistic selection strategy to balance exploration and exploitation. This hybrid approach aims to leverage the strengths of both algorithms for better convergence and broader search across the solution space.