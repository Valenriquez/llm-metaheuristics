 # Name: GravitationalSearchMetaheuristic
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
    'metropolis'
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
# The chosen metaheuristic is Gravitational Search Metaheuristic (GSM), which simulates the gravitational force between masses to search for optimal solutions in a problem space. 
# The first operator, 'gravitational_search', uses parameters gravity and alpha to influence the search behavior, with selector set to 'metropolis' for probabilistic selection.
# The second operator, 'random_flight', involves scaling factor, distribution type (levy), and beta parameter. It is configured with a probabilistic selector ('probabilistic').
# These operators are selected based on their ability to balance exploration and exploitation effectively in the optimization process using different mechanisms defined by their parameters and selectors.