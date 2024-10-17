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
# The chosen metaheuristic is Gravitational Search Optimization (GSO), inspired by the gravitational force in physics, which mimics the behavior of particles moving towards masses with higher potential energy. In this case, we use a single gravitational search operator with gravity set to 1.0 and alpha to 0.02. Additionally, we include a random flight operator that uses a levy distribution for scaling and beta of 1.5, configured to operate in probabilistic mode. These operators are selected across all iterations ('all') to explore the solution space effectively. The Rastrigin function is chosen as the benchmark problem due to its multimodal nature, requiring metaheuristic techniques to find optimal solutions efficiently.
