 # Name: GravitationalSearchOptimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator 1
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
# This code defines a metaheuristic named GravitationalSearchOptimization using two operators, gravitational search and random flight. The gravitational search operator has parameters for gravity (1.0) and alpha (0.02), while the random flight operator includes scale (1.0), distribution set to levy, and beta (1.5). Both operators are used in a probabilistic mode.
# The Sphere function is chosen as the benchmark problem with dimension 2. The metaheuristic runs for 100 iterations, and results include the best solution found by the algorithm. This setup is justified based on exploring different optimization strategies tailored to continuous domains: gravitational search for global exploration and random flight for local fine-tuning, both suitable for starting points far from the optimal solution.