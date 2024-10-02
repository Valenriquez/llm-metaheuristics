 # Name: Dynamic Swarm Metaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'swarm_dynamic',
    {
        'operator': 'swarm_dynamic',
        'parameters': {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'selector': 'greedy'
    }
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named "Dynamic Swarm Metaheuristic" is designed to optimize the Rastrigin function with two dimensions using a swarm dynamic approach. 
# The search operator 'swarm_dynamic' is chosen based on the parameters provided in the parameters_to_take.txt file, specifically focusing on the factor (0.7), self confidence (2.54), swarm confidence (2.56), version ('inertial'), and distribution type ('uniform'). 
# The selector 'greedy' is selected to guide the search process according to the specified rules from the parameters_to_take.txt file, which does not require any additional operators or selectors beyond those listed in the file.