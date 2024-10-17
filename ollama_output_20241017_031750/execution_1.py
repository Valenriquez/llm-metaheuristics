 # Name: GravitationalSearchOptimizer
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
# The GravitationalSearchOptimizer (GSO) is a metaheuristic inspired by the principles of gravitational force and mass interactions. In this implementation, we use the 'gravitational_search' operator with predefined parameters for gravity and alpha. The selector is set to 'all', indicating that all elements should be considered in each iteration, which aligns with the explorative nature of GSO. We chose these settings based on typical parameter values recommended for gravitational search optimization.
# The Gravitational Search Operator uses a force-based approach where particles interact under the influence of gravity and mass. The 'gravity' parameter controls the strength of this interaction, while 'alpha' influences the scaling factor applied to the forces. These parameters were chosen as they are typical for GSO, with the 'gravity' set to 1.0 which is a standard value representing full gravitational force, and 'alpha' set at 0.02 which balances between exploration and exploitation according to the literature.
# The selector 'all' allows every element in the population to be influenced by gravity, promoting a comprehensive search across the solution space. This setting is appropriate for exploring diverse regions of the fitness landscape to find potential optimal solutions.