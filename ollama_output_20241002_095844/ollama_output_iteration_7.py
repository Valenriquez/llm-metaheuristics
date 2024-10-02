 # Name: CustomSwarmDynamic
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
# The metaheuristic named CustomSwarmDynamic is designed to optimize the Rastrigin function with two dimensions using a swarm dynamic algorithm. This algorithm is chosen based on its ability to handle nonconvex, nondifferentiable, and possibly discontinuous objective functions through a population-based search approach. 
# The swarm_dynamic operator is configured with specific parameters: factor set to 0.7, self_conf to 2.54, swarm_conf to 2.56, version as inertial, and distribution as uniform. These settings are based on typical configurations for exploring diverse solution spaces in global optimization problems.
# The selector used is 'metropolis', which follows the rule of using probabilistic selection methods suitable for metaheuristic frameworks. Since the operator involves a genetic component (swarm_dynamic), it requires a corresponding selector that supports probabilistic exploration, ensuring diversity and adaptability in the search process.