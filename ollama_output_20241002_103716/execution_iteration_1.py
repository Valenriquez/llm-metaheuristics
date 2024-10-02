 # Name: CustomSwarmMetaheuristic
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
        'distribution': 'gaussian'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed to optimize the Rastrigin function with two dimensions using a swarm dynamic approach. The search operator 'swarm_dynamic' is configured with specific parameters including factor, self_conf, swarm_conf, version, and distribution. The factor determines the influence of individual particles on their neighborhood, self_conf and swarm_conf set the cognitive and social components of particle behavior respectively. The version is set to inertial which allows for a balance between exploration and exploitation. The distribution type is Gaussian, promoting diversity in the search space. The selector 'probabilistic' uses probabilistic criteria to decide whether to accept new solutions, enhancing the explorative nature of the algorithm. This combination aims to efficiently navigate the complex landscape of the Rastrigin function while maintaining a balance between exploration and exploitation.