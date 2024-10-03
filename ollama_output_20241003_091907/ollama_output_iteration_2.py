 # Name: RandomSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'gaussian',
        {
            'parameter1': 0.5,
            'parameter2': 0.8,
        },
        'greedy'
    ),
    ( # Search operator 2
        'levy',
        {
            'parameter1': 0.3,
            'parameter2': 0.6,
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed as a Random Search Metaheuristic, utilizing two search operators: gaussian and levy. The gaussian operator uses parameters parameter1 set to 0.5 and parameter2 set to 0.8, while the levy operator employs similar parameters with values of 0.3 and 0.6 respectively. Both operators are applied using different selectors; gaussian with 'greedy' selector and levy with 'all' selector. The choice of these parameters is based on typical settings for such metaheuristics in optimization problems, aiming to explore the solution space effectively while considering both exploration (gaussian) and exploitation (levy).
# The template provided was used as a base structure, ensuring all specified operators and parameters from the parameters_to_take.txt file are included. The use of 'greedy' selector with gaussian and 'all' selector with levy is justified by their typical roles in random search algorithms: greedy for exploitation of current best solutions, and 'all' for broader exploration across the solution space.
