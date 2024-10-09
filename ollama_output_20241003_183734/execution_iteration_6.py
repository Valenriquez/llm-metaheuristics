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
# The metaheuristic chosen is Gravitational Search Optimization (GSO), which mimics the gravitational pull between masses in a universe to explore the search space. Two key operators are implemented here: 
# 1) Gravitational Search, with parameters gravity set to 1.0 and alpha to 0.02, affecting the strength of the gravitational force, and selector set to 'all' for applying this operator globally across the population.
# 2) Random Flight, using a scale of 1.0 and distribution set to 'levy', which represents random movements influenced by Levy flights characteristic in foraging behavior of birds or insects. The selector is set to 'probabilistic' to allow these movements with probability.
# These operators are chosen based on the ability to balance exploration (gravitational search) and exploitation (random flight) required for global optimization tasks, as supported by the parameters specified according to literature references from parameters_to_take.txt.