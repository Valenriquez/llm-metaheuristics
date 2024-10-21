 # Name: GravitationalSearchOptimization
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'greedy'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a metaheuristic named GravitationalSearchOptimization using the Gravitational Search algorithm, which is inspired by the laws of gravity and physics. 
# The first operator in 'gravitational_search' uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, with a selector type 'greedy'. This setting aims to simulate the gravitational force within the search space for better exploration.
# The second operator 'random_flight' employs a random flight pattern using Levy distribution for exploration, with parameters 'scale' set to 1.0 and 'beta' fixed at 1.5. It uses the selector type 'all', indicating that this method should be applied in every iteration of the search process.
# These operators are designed to balance between exploration (random flight) and exploitation (gravitational search) within the solution space, which is crucial for optimization problems like the Rastrigin function used here. The combination of greedy selection from gravitational search and random exploration based on Levy flights should lead to effective convergence towards an optimal solution.