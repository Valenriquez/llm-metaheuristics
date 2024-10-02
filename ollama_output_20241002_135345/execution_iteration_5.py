 # Name: CustomSearchMetaheuristic
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
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'uniform'
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is named CustomSearchMetaheuristic and utilizes a swarm dynamic search operator. 
# The parameters for the swarm_dynamic include a factor of 0.7, self_conf set to 2.54, swarm_conf to 2.56, version as inertial with distribution set to uniform. 
# The selector used is metropolis which is appropriate for probabilistic selection based on probability distributions. 
# This setup aims to balance exploration and exploitation in the optimization process by adjusting the influence of individual particles' cognitive and social components dynamically.