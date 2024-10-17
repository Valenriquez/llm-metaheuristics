 # Name: MyCustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
            ( # Search operator 1
            'random_flight',
            {
                'scale': 0.5,
                'distribution': 'levy',
                'beta': 1.0
            },
            'probabilistic'
            ),
            (  
            'local_random_walk',
            {
                'probability': 0.8,
                'scale': 0.5,
                'distribution': 'gaussian'
            },
            'all'
        )
      ]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named MyCustomMetaheuristic is designed to solve the Rastrigin function with two dimensions using a combination of random flight and local random walk search operators. 
# The random flight operator is configured with a scale of 0.5, distribution set to levy, and beta value of 1.0. This helps in exploring the solution space by following a levy distribution which can help avoid local minima.
# The local random walk operator is set to operate on all possible solutions with an increased probability (0.8) and scale of 0.5. It uses Gaussian distribution for mutation, which tends to explore more diverse areas of the search space.
# Both operators are combined in a probabilistic framework where the choice between exploration and exploitation depends on their respective probabilities and configurations. The overall approach aims to balance between exploring new regions of the function landscape and refining solutions through local improvements.