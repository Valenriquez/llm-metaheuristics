 # Name: CustomMetaheuristic
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
# This metaheuristic is designed to solve the Rastrigin function optimization problem using a custom search operator called 'swarm_dynamic'. The parameters for this operator are set according to the guidelines provided in parameters_to_take.txt, specifically setting the factor to 0.7, self_conf and swarm_conf to 2.54 and 2.56 respectively, with the version as inertial and distribution as uniform. The selector used is 'metropolis', which is suitable for probabilistic selection based on acceptance criteria in optimization problems. This combination aims to balance exploration and exploitation effectively within a stochastic framework.
