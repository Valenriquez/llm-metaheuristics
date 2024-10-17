 # Name: GravitationalSearchAlgorithm
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
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The algorithm is named GravitationalSearchAlgorithm as it combines the concepts of gravitational search with a metaheuristic approach to optimize functions. 
# The Rastrigin function, which is a standard test function in optimization, is chosen for its multi-modal nature and challenge to global optimizers.
# Two main operators are used: gravitational_search and random_flight. Gravitational search emulates the gravitational force between masses to explore the solution space, while random flight uses probabilistic distribution to escape local minima.
# The parameters for each operator are set according to their typical values in literature or adjusted based on empirical tests found in parameters_to_take.txt: 
# - gravity (1.0) and alpha (0.02) are standard settings for gravitational search, enhancing exploration by simulating the influence of mass on spatial distribution.
# - random flight uses a scale factor (1.0), distribution type ('levy') which is known to balance between exploitation and exploration, and beta (1.5) that influences the direction change probability in a Levy-flight like pattern.
# Selectors are set as 'greedy' for immediate optimization focus and 'probabilistic' for a chance-driven approach, respectively. This setup aims to leverage both deterministic and probabilistic strategies to find an optimal solution efficiently.