Here's a complete example of how you could implement a metaheuristic:

      
# Name: Random Search
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.random_search({5}) 
prob = fun.get_formatted_problem()

heur = [
    (
        'Random Flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'greedy'
    ),
    (
        'Genetic Crossover',
        {
            'pairing': 'rank',
            'crossover': 'single',
            'mating_pool_factor': 0.4
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# This metaheuristic uses two search operators: Random Flight and Genetic Crossover.
# The selection of the parameters for these operators is randomized during each iteration,
# allowing the algorithm to adapt to the best combination of parameters in a certain range.

# The Random Search function from benchmark_func serves as the benchmark function used for the optimization process.
# The number of dimensions was set to 5, which is used throughout the metaheuristic.