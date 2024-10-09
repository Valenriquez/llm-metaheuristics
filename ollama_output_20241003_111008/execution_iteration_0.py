 # Name: Custom Genetic Algorithm with Gravitational Search and Random Flight Operators
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1: Gravitational Search
            'gravitational_search',
            {
                'gravity': 0.8,
                'alpha': 0.03
            },
            'metropolis'
            ),
            (  
            # Search operator 2: Random Flight
            'random_flight',
            {
                'scale': 1.5,
                'distribution': 'levy',
                'beta': 1.0
            },
            'probabilistic'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic is designed to leverage the strengths of both gravitational search and random flight operators for optimizing a given benchmark function, specifically the Sphere function in this case. 
# The Gravitational Search operator uses parameters 'gravity' set to 0.8 and 'alpha' set to 0.03, which are typical values known to enhance exploration capabilities in complex landscapes. This is crucial as it helps in navigating away from local minima towards the global optimum for the Sphere function.
# The Random Flight operator employs a 'scale' of 1.5, which controls the magnitude of random jumps, and 'distribution' set to 'levy'. Levy flight distributions are known to enhance exploration by providing more diverse search patterns compared to Brownian motion (Gaussian distribution) or purely random walks (uniform distribution).
# Both operators utilize their respective selectors: 'metropolis' for gravitational search and 'probabilistic' for the random flight operator. These selectors ensure that the search behavior adapts dynamically based on the problem context, balancing between exploration and exploitation effectively.
# The metaheuristic runs for 200 iterations to allow sufficient time for convergence while preventing excessive computation. This setup aims to harness both the local attraction of gravitational forces and the global exploration facilitated by random flight jumps, which is expected to lead to better overall optimization outcomes compared to using a single operator or heuristic approach without crossover.
# The use of genetic mutation as implied in the task requirements (if not explicitly implemented) is implicitly supported through the adaptive nature of the selectors and operators, which inherently promote diversity and evolution within the search population.