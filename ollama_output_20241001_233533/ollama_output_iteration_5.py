 # Name: Custom Metaheuristic
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
# The metaheuristic chosen is named "Custom Metaheuristic." It utilizes a swarm dynamic optimization algorithm with parameters designed to explore the solution space effectively. The factor is set to 0.7, which influences the scaling of the particle's velocity. Self-confidence (self_conf) and swarm confidence (swarm_conf) are both set to 2.54 and 2.56 respectively, affecting how much each particle trusts itself versus the group dynamics. The version is inertial, promoting a balance between exploration and exploitation by maintaining the current velocity component in the next position update. The distribution is Gaussian, which introduces randomness proportional to the standard deviation of the particles' positions, aiding in diverse search across the problem space.
# The selector chosen is "metropolis," which is appropriate for probabilistic decisions based on acceptance criteria derived from energy-like functions often encountered in optimization problems. This selector fits well with the stochastic nature of our metaheuristic and ensures that only beneficial moves are accepted, leading to better convergence properties in complex landscapes.