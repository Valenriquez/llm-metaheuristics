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
# The chosen metaheuristic is named "Dynamic Swarm Metaheuristic." 
# It utilizes a swarm dynamic search operator with specific parameters to optimize the Rastrigin function, which is suitable for continuous optimization problems.
# The factor is set to 0.7, self_conf to 2.54, and swarm_conf to 2.56 are chosen based on typical settings for such operators.
# Inertial version of the dynamic movement is selected with a uniform distribution for exploring the search space.
# The greedy selector is used as it efficiently guides the search towards better solutions in each iteration.
