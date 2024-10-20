 # Name: GravitationalSearchAlgorithm
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(22)
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
# The Gravitational Search Algorithm (GSA) is a metaheuristic inspired by the principles of gravitational force and mass interaction. In this implementation, we use the 'gravitational_search' operator with specific parameters: gravity set to 1.0 and alpha to 0.02. The selector used is 'all', which means all particles in the population will undergo a gravitational search at each iteration. This approach aims to mimic the behavior of celestial bodies under gravitational influence, promoting exploration and exploitation through mass interactions that are governed by their gravitational forces.
# The Sphere function (as represented by bf.Sphere(22)) is chosen as the benchmark problem here, which has no local minima and is continuous, making it suitable for testing optimization algorithms without complications from discontinuities or multiple minima.
# GSA uses a set of parameters that balance between global and local search capabilities. The 'gravity' parameter controls the strength of gravitational force, while 'alpha' influences the rate at which solutions converge to better regions. This setup allows for an adaptive exploration-exploitation trade-off during optimization.