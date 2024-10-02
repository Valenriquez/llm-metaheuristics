 # Name: PSOwithAdaptiveSwarmDynamic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'Particle Swarm Optimization',
    {
        'swarm_dynamic': {
            'operator': 'Inertial',
            'distribution': 'Gaussian'
        },
        'selector': 'Metropolis'
    },
    'All'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# PSO is a metaheuristic inspired by the behavior of bird flocks or fish schools, where individuals in the group move collectively towards areas of higher fitness. The adaptive swarm dynamic uses a Gaussian distribution to adjust particle velocities adaptively, which allows for better exploration and exploitation of the search space. The Metropolis selector helps balance between exploration (randomness) and exploitation (focus on promising regions), making it suitable for global optimization problems where both aspects are crucial. This combination aims to leverage the strengths of both PSO and its adaptive dynamics while ensuring a diverse exploration through the selection criteria, thus enhancing overall performance in optimizing complex non-convex functions like Rastrigin's.