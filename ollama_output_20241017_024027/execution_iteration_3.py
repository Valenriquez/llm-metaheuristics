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
# The algorithm GravitationalSearchOptimization is used with parameters gravity set to 1.0 and alpha to 0.02 for the gravitational search operator, and random flight with scale of 1.0, distribution as levy, and beta of 1.5 with a probabilistic selector. These settings are chosen based on typical values found in literature for similar metaheuristic algorithms and parameters that have shown effectiveness across various benchmark functions. The GravitationalSearchOptimization is designed to mimic the gravitational forces between masses to search for optimal solutions by iteratively adjusting positions according to predefined rules. The selection of Levy distribution in random flight adds a probabilistic element, promoting exploration beyond local minima. The 'all' selector allows all possible operators to be applied at each step, enhancing the diversity and adaptability of the search process.
