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
# The Gravitational Search Algorithm (GSA) is a population-based metaheuristic inspired by the gravitational force in physics. 
# In this implementation, we use GSA with parameters gravity set to 1.0 and alpha to 0.02. This configuration aims to simulate the effect of gravity on particles within a search space.
# The algorithm includes both greedy and probabilistic selection strategies as specified by the selectors 'all' and 'probabilistic'.
# Additionally, we use Random Flight operator with scale set to 1.0 and distribution type set to levy. This helps in exploring the solution space more effectively through random movements.
# Both operators are carefully chosen based on their ability to balance exploration and exploitation within the optimization process. The GSA is known for its robustness and effectiveness in handling complex non-linear and multi-modal functions, which makes it suitable for benchmarking problems like Rastrigin function here.