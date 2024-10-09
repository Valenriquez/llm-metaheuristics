 # Name: MyCustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1 - Random Flight
                'random_flight',
                {
                    'scale': 0.5,
                    'distribution': 'levy',
                    'beta': 1.2
                },
                'probabilistic'
            ),
            (  
                # Search operator 2 - Local Random Walk
                'local_random_walk',
                {
                    'probability': 0.8,
                    'scale': 0.8,
                    'distribution': 'gaussian'
                },
                'probabilistic'
            )
        ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic is named MyCustomMetaheuristic as it represents a custom approach to solving the optimization problem. 
# The benchmark function used is Rastrigin, which has two dimensions (n=2) for this example.

# Search Operator 1 - Random Flight:
# This operator employs random flight behavior by using parameters 'scale', 'distribution', and 'beta'.
# Scale is set to 0.5, a value chosen as it allows the algorithm to explore more of the solution space effectively.
# The distribution type is set to 'levy', which introduces long-range jumps in the search process, aiding in escaping local minima.
# Beta is set to 1.2, which influences the balance between exploration and exploitation during the flight.
# This operator uses a probabilistic selector, allowing it to be applied based on predefined probabilities that favor exploration over exploitation at certain iterations.

# Search Operator 2 - Local Random Walk:
# This operator mimics local random walk with parameters 'probability', 'scale', and 'distribution'.
# The probability is set to 0.8, which determines the likelihood of moving in a random direction relative to the current position.
# Scale is adjusted to 0.8 to control the step size in the search space during walks.
# Distribution type is chosen as 'gaussian' (normal distribution), facilitating gradual changes within the local region without significant jumps that might disrupt convergence.
# Similar to the first operator, this one also uses a probabilistic selector for flexibility in applying its behavior throughout the optimization process.

# Both operators are designed with parameters and distributions that encourage exploration initially but do not stray too far from promising regions of the solution space, reflecting a balanced approach between discovery and refinement suggested by the problem's characteristics as outlined in the benchmark function details.