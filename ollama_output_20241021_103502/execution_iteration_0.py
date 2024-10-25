 # Name: GravitationalSearchOptimizationWithRandomFlight
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
# The GravitationalSearchOptimizationWithRandomFlight metaheuristic combines the gravitational search algorithm with a random flight operator for exploration. 
# The gravitational search utilizes gravity and alpha parameters to simulate the attraction-repulsion forces among particles, promoting convergence towards better solutions. 
# The random flight operator introduces diversity by scaling the step size according to a distribution (levy in this case), which helps in escaping local minima.
# The probabilistic selector is used for the random flight operator to ensure stochastic exploration while balancing with gravitational attraction for better coverage of the search space.
