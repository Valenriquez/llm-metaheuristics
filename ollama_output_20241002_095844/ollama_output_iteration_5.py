 # Name: CustomSwarmDynamicMetaheuristic
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
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The custom metaheuristic is named CustomSwarmDynamicMetaheuristic, utilizing a swarm dynamic algorithm with specified parameters for exploration in the search space. 
# The Rastrigin function (dimension 2) is chosen as it's suitable for benchmarking global optimization problems.
# The swarm_dynamic operator is configured with factor 0.7, self_conf set to 2.54, swarm_conf to 2.56, version 'inertial', and distribution type 'uniform'.
# A probabilistic selector ensures diversity in the search process, promoting exploration over exploitation.
# The metaheuristic runs for 100 iterations with verbose output enabled to monitor the optimization progress.