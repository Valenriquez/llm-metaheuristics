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
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic, which uses the gravitational search algorithm. The parameters for the search operator include 'gravity' set to 1.0 and 'alpha' set to 0.02. The selector is set to 'all', meaning all possible selections of candidate solutions will be considered during the search process.
# The benchmark function used here is Rastrigin with a dimension of 2, which is appropriate for continuous optimization problems where the goal is to minimize the sum of squared errors between the computed and the actual values of each variable.
# The gravitational search algorithm is inspired by the physics concept of gravity affecting masses in a multidimensional space. It involves applying forces on candidate solutions based on their fitness scores (lower scores are attracted more strongly) and adjusting their positions according to these forces, allowing for exploration of different areas of the solution space.
# By setting 'gravity' high, we emphasize the influence of better-performing candidates, encouraging convergence towards a global minimum. The parameter 'alpha' controls the step size in each iteration, influencing how aggressively the algorithm explores the search space.
# This setup aims to balance between exploration and exploitation by allowing all possible selections during the search, which can help in finding more diverse solutions while still refining them based on their fitness scores.