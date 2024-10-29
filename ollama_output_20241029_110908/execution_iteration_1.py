 # Name: GravitationalSearchAlgorithmMetaheuristic
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
# The code defines a metaheuristic named GravitationalSearchAlgorithmMetaheuristic, which incorporates two operators from the parameters provided in parameters_to_take.txt. 
# The first operator is gravitational_search with gravity set to 1.0 and alpha to 0.02. This search operator will use all available strategies ('all') for exploring the solution space.
# The second operator is random_flight, which involves a scale of 1.0, a distribution type 'levy', and beta of 1.5. This operator uses a probabilistic strategy ('probabilistic') to navigate through possible solutions.
# These operators are applied iteratively for a total of 100 iterations to find the best solution according to the Rastrigin function benchmarked in this setup. The verbose setting is enabled to provide detailed output during the optimization process, including the final best solution found.