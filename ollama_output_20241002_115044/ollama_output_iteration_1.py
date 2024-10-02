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
# The chosen metaheuristic is named CustomMetaheuristic. It utilizes a swarm dynamic search operator which includes parameters for factor, self confidence, swarm confidence, version, and distribution. These parameters are set based on the provided template to ensure effective exploration of the solution space. The selector used in this implementation is Metropolis, which is appropriate given the nature of global optimization problems where nonconvex and possibly discontinuous functions need exploring.
# The Rastrigin function is selected as it is a standard benchmark for testing metaheuristic algorithms due to its multiple local minima, making it suitable for evaluating the performance of the search operator in handling complex landscapes.
# The swarm_dynamic operator with specified parameters and the Metropolis selector are implemented following the guidelines provided, ensuring that all parameters and operators used are compliant with the constraints outlined in the parameters_to_take.txt file.