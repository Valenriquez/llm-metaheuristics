 # Name: Custom Metaheuristic
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
        'factor': [0.7, 1.0],
        'self_conf': [2.54],
        'swarm_conf': [2.56],
        'version': ['inertial', 'constriction'],
        'distribution': ['uniform', 'gaussian', 'levy']
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses a dynamic swarm optimization approach with multiple configurations for the search distribution. The factor parameter is varied between 0.7 and 1.0 to balance exploration and exploitation, while self_conf and swarm_conf are set to specific values to tune the influence of individual particles on their neighborhood. The version can be either inertial or constriction, allowing for different dynamics within the swarm. Distributions include uniform, gaussian, and levy, which offer flexibility in how the search space is explored. This setup aims to leverage diverse exploration techniques to better handle complex nonconvex landscapes typical in ill-structured global optimization problems.
