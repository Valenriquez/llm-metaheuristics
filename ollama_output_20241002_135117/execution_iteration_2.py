 # Name: CustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operator with parameters from parameters_to_take.txt
heur = [( # Search operator 1
    'swarm_dynamic',
    {
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'uniform'
    },
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is named CustomMetaheuristic. It uses the Swarm Dynamic search operator with specific parameters to optimize a problem using the Rastrigin function, which has two dimensions (specified in fun = bf.Rastrigin(2)). The swarm_dynamic operator is configured with a factor of 0.7, self_conf set to 2.54, swarm_conf to 2.56, version set to inertial, and distribution type as uniform. The selector used is greedy for this optimization task. This selection ensures that the algorithm focuses on exploring promising regions in the search space based on the current best solutions found by individual agents within the swarm.