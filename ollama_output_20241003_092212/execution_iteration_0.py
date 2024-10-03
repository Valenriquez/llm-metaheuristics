 # Name: Custom Metaheuristic with Adaptive Operators
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1: Gravitational Search Optimization with Metropolis Selector
            'gravitational_search',
            {
                'gravity': 0.8,
                'alpha': 0.02
            },
            'metropolis'
            ),
            (  
            # Search operator 2: Random Flight with Probabilistic Selector
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

# Short explanation and justification:
# This metaheuristic is designed to explore the Sphere benchmark function using a combination of two adaptive search operators, each with its specific parameters and selection method. 
# The first operator, Gravitational Search Optimization (GSO), employs a gravitational force model inspired by physical principles to guide the search towards better solutions. With a lower gravity value (0.8) compared to typical settings, it encourages more diverse exploration. The alpha parameter is set to 0.02 to control the scaling of accelerations due to mass differences. This operator uses the Metropolis selector for decision-making during iterations.
# The second operator, Random Flight, simulates random movements influenced by a Levy distribution and a scaling factor (1.5). This allows for both local and global explorative movements within the search space, with a bias towards longer jumps characteristic of Levy flights. The probabilistic selector is chosen to determine whether these flights lead to an improvement in the solution based on predefined probabilities.
# Both operators are designed to complement each other: GSO with its diverse exploration capabilities aids in finding promising areas, while Random Flight enhances the ability to escape local minima by allowing more extensive search trajectories. This combination aims to balance between exploitation and exploration effectively across multiple iterations.