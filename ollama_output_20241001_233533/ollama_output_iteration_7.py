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
# This metaheuristic uses a swarm dynamic optimization algorithm with specific parameters for exploration and exploitation. The 'swarm_dynamic' operator is configured with an inertial version, Gaussian distribution, and a factor of 0.7 to balance the influence of cognitive and social components in the particle movement. The selector used is 'metropolis', which is suitable for probabilistic decision-making processes in optimization scenarios. This approach aims to navigate through the search space efficiently by adapting its exploration based on local information gathered from each particle's experience, influenced by both its personal best and the swarm's collective knowledge.
