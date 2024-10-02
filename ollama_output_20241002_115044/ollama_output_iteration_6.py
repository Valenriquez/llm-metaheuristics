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
        'distribution': 'gaussian'
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is named CustomMetaheuristic. It uses a swarm dynamic search operator with specified parameters including factor, self confidence, swarm confidence, version, and distribution type. The selector used here is Metropolis, which is appropriate for probabilistic selection in the context of swarm dynamics. This setup is justified for optimizing Rastrigin function due to its global optimization nature and nonconvex behavior typical of benchmark functions requiring metaheuristic approaches. The parameters are chosen based on standard practices for similar algorithms as per parameters_to_take.txt, ensuring adherence to recommended values and types for each parameter.