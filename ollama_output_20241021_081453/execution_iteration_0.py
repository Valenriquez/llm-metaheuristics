 # Name: GravitationalSearchOptimizer
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
# The chosen metaheuristic is Gravitational Search Optimizer (GSO), which emulates the gravitational force between masses to search for optimal solutions. 
# In this implementation, we use two operators: gravitational_search and random_flight. 
# The gravitational_search operator uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, aiming to simulate the attraction and repulsion forces between particles in a multi-dimensional space. 
# The selector for this operator is set to 'greedy', which means it will predominantly use greedy selection strategies during optimization.
# The random_flight operator uses parameters 'scale' set to 1.0, 'distribution' set to 'levy', and 'beta' set to 1.5. 
# This operator represents a random flight behavior mimicking the foraging habits of animals following chaotic paths, which is particularly useful in explorative searches. 
# The selector for this operator is set to 'probabilistic', indicating that its movements are governed by probabilistic rules allowing it to balance between exploration and exploitation during optimization.
# Together, these operators leverage different strategies—one being deterministic (greedy selection) and the other stochastic (probabilistic selection)—to effectively navigate the search space of the Rastrigin function, a benchmark problem commonly used in testing metaheuristic algorithms.