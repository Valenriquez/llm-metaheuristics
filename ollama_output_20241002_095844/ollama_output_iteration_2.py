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
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The name of the metaheuristic is CustomSwarmDynamicMetaheuristic, inspired by swarm dynamics with a dynamic adjustment mechanism. 
# We chose the Rastrigin function as it is a good test case for non-convex functions often encountered in global optimization problems.
# The search operator is set to 'swarm_dynamic' with specific parameters including factor (0.7), self_conf (2.54), swarm_conf (2.56), version ('inertial'), and distribution ('uniform'). These settings are based on typical configurations for exploring diverse regions in the solution space.
# The selector used is 'metropolis' which fits well with probabilistic exploration in metaheuristics, ensuring a balance between exploitation and exploration as required.
# This setup aims to leverage the strengths of swarm dynamics in navigating complex landscapes while employing a selective mechanism that balances convergence speed and diversity preservation.