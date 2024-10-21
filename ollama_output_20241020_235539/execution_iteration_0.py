 # Name: GravitationalSearchAlgorithmMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements the Gravitational Search Algorithm (GSA), a metaheuristic inspired by the laws of gravity and mass interactions. The algorithm is designed to mimic the behavior of celestial bodies under gravitational forces, where more massive objects attract lighter ones. In the context of optimization, this translates to attracting solutions towards better regions in the search space.
# The GSA utilizes two primary parameters: 'gravity' (representing the strength of the gravitational force) and 'alpha' (a scaling factor that affects how quickly solutions converge). These parameters are set according to the specifications from 'parameters_to_take.txt'.
# The selector used in this implementation is 'all', indicating that all particles or candidate solutions participate in the search process, mimicking a swarm behavior where each entity influences and is influenced by others within its gravitational field. This approach encourages exploration of diverse areas in the solution space, promoting both convergence to an optimal region and escape from local minima.
# The Rastrigin function, chosen as the benchmark problem, has a domain on the real-valued space [-5.12, 5.12] for two dimensions. It is known for having multiple global minima, making it suitable for testing optimization algorithms that can handle multimodal landscapes.
# By setting 'gravity' to 1.0 and 'alpha' to 0.02, the algorithm balances between thorough exploration of the search space and exploitation of potentially promising areas identified during initial iterations. The use of 'all' as the selector ensures a comprehensive search across all candidate solutions, which is particularly beneficial for exploring complex, multimodal functions like the Rastrigin function.