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
# The algorithm chosen is a custom metaheuristic inspired by swarm dynamics, which is designed for continuous optimization problems such as the Rastrigin function in this case. The 'swarm_dynamic' operator is configured with specific parameters including factor (0.7), self_conf (2.54), swarm_conf (2.56), version ('inertial'), and distribution ('uniform'). These settings are based on empirical findings that favor a balance between exploration and exploitation in the optimization process, as indicated by 'self_conf' and 'swarm_conf'. The selector used is 'metropolis', which combines elements of both greedy and probabilistic selection to guide the search towards better solutions while allowing for occasional random jumps. This approach aims to leverage the strengths of swarm intelligence algorithms with a focus on balancing exploration and exploitation, typical in many optimization tasks.