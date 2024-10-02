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
        'distribution': 'gaussian'
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed to optimize the Rastrigin function with two dimensions using a custom search operator called 'swarm_dynamic'. The parameters for this operator are set according to best practices from the given template, specifically focusing on configuring the swarm dynamics for exploration in the optimization space. The selector used here is 'metropolis', which implements probabilistic selection based on the Metropolis-Hastings algorithm, suitable for problems where both exploration and exploitation are crucial. This combination aims to balance between exploring new areas of the search space and refining solutions within promising regions, leveraging the adaptive properties of the swarm dynamics and the probabilistic selection criteria provided by the metropolis selector.