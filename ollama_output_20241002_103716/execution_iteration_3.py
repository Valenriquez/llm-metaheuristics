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
# The metaheuristic is named CustomMetaheuristic as it combines elements from various heuristics to create a unique approach tailored for global optimization problems. We start by defining the benchmark function Rastrigin with dimension 2, which is suitable for continuous domains and aims to minimize noise and nonlinearity.
# For the search operator, we chose 'swarm_dynamic' inspired by swarm intelligence algorithms that can handle nonconvex and ill-structured problems well. The parameters are set as follows:
# - 'factor': 0.7 balances the influence of cognitive (personal best) and social (global best) components in updating particle positions.
# - 'self_conf' and 'swarm_conf' are set to 2.54 and 2.56 respectively, which control the scaling of the personal and global best position updates. The values chosen reflect a balance between exploration and exploitation.
# - 'version' is set to 'inertial', indicating that the algorithm includes an inertial term to preserve the particle's previous velocity for better convergence properties.
# - 'distribution' is set to 'uniform', which means particles are initialized with random positions within the search space, promoting a diverse exploration of the solution space.
# The selector used here is 'metropolis', which follows probabilistic criteria in selecting new states, useful for handling complex and noisy optimization problems by probabilistically accepting worse solutions to escape local minima.
# This setup aims to balance between exploring uncharted territories and exploiting known promising areas, optimizing both exploration and exploitation through a combination of cognitive-based and swarm intelligence principles.