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
# The chosen metaheuristic is inspired by the Gravitational Search Algorithm (GSA), which mimics the gravitational force between masses in a multidimensional space to search for optimal solutions. 
# For the first operator, 'gravitational_search', it uses parameters gravity and alpha to simulate the gravitational forces within the search space, with selector set to 'metropolis' to encourage probabilistic exploration.
# The second operator, 'random_flight', employs a random flight mechanism using parameters scale, distribution (set as 'levy'), and beta. This is intended for exploring diverse areas of the solution space with a bias towards Levy distributions that can offer more flexibility in search paths.
# Both operators are chosen based on their ability to balance exploration and exploitation, which is crucial for effective optimization across different types of objective functions, including those modeled by Rastrigin function here.