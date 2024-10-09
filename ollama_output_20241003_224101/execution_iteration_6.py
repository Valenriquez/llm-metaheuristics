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
# The Gravitational Search Algorithm (GSA) is a metaheuristic inspired by the laws of gravity and mass interactions. In this implementation, we use 'gravitational_search' with parameters 'gravity' set to 1.0 and 'alpha' to 0.02. This operator will primarily be used in combination with a 'metropolis' selector, which is appropriate for probabilistic selection methods.
# Additionally, we introduce the 'random_flight' operator, designed to explore solutions using a random walk characterized by its distribution type ('levy') and scaling factor ('beta'). This operator will also operate under a 'probabilistic' selector that allows for stochastic decision-making based on probability distributions.
# Both operators are selected from predefined lists in the parameters_to_take.txt file, ensuring compliance with specified requirements and maintaining consistency in metaheuristic application.