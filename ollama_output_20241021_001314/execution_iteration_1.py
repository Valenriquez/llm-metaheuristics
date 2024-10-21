 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

gravitational_gravity = 1.0
gravitational_alpha = 0.02
selector = "probabilistic"

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': gravitational_gravity,
            'alpha': gravitational_alpha
        },
        selector
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic, which uses the gravitational search algorithm to solve an optimization problem. 
# The algorithm is configured with specific parameters for gravity and alpha as defined in the parameters_to_take.txt file. 
# The selector parameter is set to "probabilistic" to ensure that the probabilistic selection method is used during the search process. 
# This setup allows the gravitational search operator to explore the solution space efficiently, leveraging both a predefined gravity constant and an alpha coefficient for fine-tuning the search behavior.
