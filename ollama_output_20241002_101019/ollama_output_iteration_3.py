 # Name: Adaptive Swarm Dynamic Optimization with Inertial Constriction Factor
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Search operator using probabilistic selection based on the constriction factor version of the inertial dynamic swarm optimization
heur = [(
    'swarm_dynamic',
    {
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'constriction',
        'distribution': 'gaussian'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named "Adaptive Swarm Dynamic Optimization with Inertial Constriction Factor" is designed to optimize the Rastrigin function, which is a standard test problem for optimization algorithms due to its nonconvex nature. 
# The algorithm uses a swarm_dynamic operator with specific parameters: factor set to 0.7, self_conf and swarm_conf both set to 2.54 and 2.56 respectively, version selected as constriction, and distribution chosen as gaussian. 
# This setup is based on the idea of enhancing exploration and exploitation through a dynamic adjustment mechanism typical in swarm intelligence algorithms. The probabilistic selection method ensures that decisions are made probabilistically, which can lead to better convergence properties in complex landscapes.
# The selector 'probabilistic' is used to guide the search by choosing actions with probabilities corresponding to their expected utility or goodness, as opposed to deterministic choices based on complete information about the state of the system. This approach allows for a balance between exploration and exploitation, which is crucial for handling diverse and potentially conflicting objectives in complex optimization problems.
