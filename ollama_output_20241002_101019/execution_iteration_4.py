 # Name: Dynamic Swarm Probabilistic Search
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Search operator using probabilistic selection with Gaussian distribution for inertial swarm dynamics
heur = [(
    'swarm_dynamic',
    {
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'gaussian'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic chosen is named "Dynamic Swarm Probabilistic Search." 
# It is designed to optimize the Rastrigin function with two dimensions, which is a standard benchmark for testing optimization algorithms due to its multimodal nature.
# A search operator using probabilistic selection ('probabilistic') is employed, specifically tailored for dynamic swarm behavior. The swarm dynamics are configured with a Gaussian distribution ('distribution': 'gaussian'), ensuring a balance between exploration and exploitation by considering the inertial component of the swarm's movement. This is controlled through the 'factor' parameter which influences the weight given to the current velocity.
# Additionally, parameters such as self_confidence (self_conf) and swarm_confidence (swarm_conf) are set according to typical values observed in literature for similar optimization scenarios. The version of the dynamics ('inertial') is selected based on empirical evidence that it can effectively handle continuous search spaces.