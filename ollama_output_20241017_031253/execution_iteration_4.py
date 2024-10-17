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
# The metaheuristic chosen is Gravitational Search Algorithm (GSA), which simulates the gravitational force between masses to search for the optimal solution. 
# The first operator used is 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02, selected by 'metropolis' selector. 
# This operator uses a probabilistic approach to explore the solution space by simulating gravitational forces between masses in a multi-dimensional space.
# The second operator is 'random_flight' with scale set to 1.0 and distribution type as 'levy'. It employs a probabilistic selection strategy ('probabilistic') for exploration. 
# This method involves random flights that can be influenced by the levy flight distribution, allowing for both local and global explorations within the search space.
# Together, these operators use a combination of gravitational attraction (exploitation) and random flight mechanisms (exploration), aiming to converge towards an optimal solution efficiently.