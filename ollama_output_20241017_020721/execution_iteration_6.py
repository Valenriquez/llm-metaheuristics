 # Name: GravitationalSearchMetaheuristic
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
# This metaheuristic is named GravitationalSearchMetaheuristic as it uses the gravitational search algorithm, which is characterized by parameters such as gravity and alpha to guide its search process. The gravitational search operator is configured with a fixed value of gravity set to 1.0 and an alpha coefficient of 0.02. This configuration aims to balance between exploration and exploitation in the optimization process, where higher gravity values promote more global exploration while lower values focus on local exploitation.
# The selector 'all' means that the gravitational search operator will be applied to all possible solutions during each iteration of the metaheuristic, allowing for a comprehensive evaluation across the entire population at every step. This approach helps in navigating through multiple peaks and valleys of the objective function more efficiently by exploring diverse regions of the search space simultaneously.
# The benchmark function used here is Rastrigin's function which is suitable for testing optimization algorithms due to its multimodal nature, having many local minima, providing a challenging testbed for metaheuristic techniques. This setup allows for a direct comparison with other similar algorithms by evaluating their performance in solving the same problem using different parameters and strategies.
# The metaheuristic runs for 100 iterations, enabling thorough exploration of the search space before reaching a final solution that is considered the best found so far after these iterations are complete. This iterative process helps in refining the solutions towards better fitness values according to the Rastrigin function's criteria while adhering to the constraints and conditions defined within the algorithm setup.