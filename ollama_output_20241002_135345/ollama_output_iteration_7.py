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
# This metaheuristic uses a custom search operator named 'swarm_dynamic' which is designed based on the parameters provided in the parameters_to_take.txt file. The 'factor', 'self_conf', 'swarm_conf', 'version', and 'distribution' parameters are set to specific values that are deemed appropriate for this type of optimization problem according to the guidelines from the benchmark function used (Rastrigin). The selector is chosen as 'metropolis' which is a probabilistic approach suitable for exploring diverse regions in the search space. This combination aims to balance exploration and exploitation, ensuring effective coverage of the solution space while avoiding premature convergence.
