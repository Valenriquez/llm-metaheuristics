 # Name: Dynamic Swarm Metaheuristic
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
        'factor': [0.7, 1.0],
        'self_conf': [2.54],
        'swarm_conf': [2.56],
        'version': ['inertial', 'constriction'],
        'distribution': ['uniform', 'gaussian', 'levy']
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The name of the metaheuristic is "Dynamic Swarm Metaheuristic". This heuristic uses a dynamic swarm optimization approach, which involves adjusting parameters such as factor, self_conf, and swarm_conf to adapt to different versions like inertial and constriction. The distribution type can be either uniform, gaussian, or levy, adding flexibility for exploring the solution space. The selector chosen is 'metropolis', which is a probabilistic selection method that decides whether to accept a new solution based on its fitness compared to the current one. This approach aims to balance exploration and exploitation through dynamic adjustments in search parameters.
