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
# This code implements a metaheuristic using the Gravitational Search Algorithm (GSA). 
# The algorithm consists of two main operators: gravitational_search and random_flight.
# 1. Gravitational search operator uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, applied to all particles.
# This is a typical setup for the gravity parameter in GSA which influences how strongly the gravitational force acts on particles.
# The alpha parameter controls the exponential decay rate of the gravitational constant, aiding convergence by reducing the influence over iterations.
# 2. Random flight operator uses 'scale' set to 1.0 and distribution type 'levy'. It is configured with a probabilistic selector ('probabilistic') meaning that its actions are chosen randomly based on probabilities.
# The levy distribution used here is known for its long tails, which can help in exploring the search space more effectively by allowing larger jumps compared to the gaussian or uniform distributions typically used in random flights.
# Both operators contribute to the exploration and exploitation balance typical of metaheuristic algorithms, ensuring a diverse range of solutions are evaluated during optimization.