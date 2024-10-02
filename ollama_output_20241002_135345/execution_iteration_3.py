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
    'metropolis'
)]

# Create and run the metaheuristic with defined parameters
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is named CustomMetaheuristic as it uses a custom search operator defined by the parameters in the provided template. The search operator used here is 'swarm_dynamic' which includes specific settings for 'factor', 'self_conf', 'swarm_conf', 'version', and 'distribution'. These are all within the allowed parameters from parameters_to_take.txt.
# The selector 'metropolis' is chosen to optimize based on probabilistic criteria, which aligns with the guidelines provided in the template regarding random search algorithms for ill-structured global optimization problems.
# This setup aims to explore a range of potential solutions using the defined settings and selecting the best based on the metropolis criterion, ensuring a balance between exploration and exploitation typical of such probabilistic approaches.