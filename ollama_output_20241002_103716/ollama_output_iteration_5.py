 # Name: CustomSwarmOptimizer
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
# The chosen metaheuristic is named CustomSwarmOptimizer. It uses a swarm dynamic search operator with specified parameters including factor, self confidence, swarm confidence, version, and distribution type. The selector used in this implementation is metropolis which combines elements of both greedy and probabilistic selection methods for better exploration and exploitation trade-offs in the optimization process. This approach aims to balance between exploring new areas of the solution space and refining solutions within promising regions, making it suitable for continuous, discrete, or mixed domains as per its application context from parameters_to_take.txt. The use of inertial version with uniform distribution enhances the algorithm's ability to adapt its search pattern dynamically according to the problem requirements and characteristics defined in the given template and constraints outlined in parameters_to_take.txt.