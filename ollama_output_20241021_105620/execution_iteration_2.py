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
    'all'
    ),
    (  
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
# The code is designed to implement a Gravitational Search Optimization (GSO) algorithm, which is inspired by the laws of gravity and mass interactions. 
# We start with a GravitationalSearch operator that uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. This operator will be applied to all possible solutions ('all').
# Additionally, we include a RandomFlight operator which utilizes a random flight strategy influenced by the Levy distribution. The scale is set to 1.0, and beta is adjusted to 1.5 for probabilistic selection.
# These operators are chosen based on their ability to explore diverse regions of the search space through gravitational attraction and random jumps respectively, which should be suitable for optimizing complex functions like Rastrigin's function.
# The algorithm runs for a predefined number of iterations (100) with verbose output enabled to track the optimization process. This setup aims to find the best solution based on the fitness evaluation of both operators.