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
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is named CustomMetaheuristic, designed to optimize the Rastrigin function with two variables using a swarm dynamic search operator. 
# The parameters for the swarm_dynamic are set as follows: factor is 0.7, self_conf is 2.54, and swarm_conf is 2.56. These values were selected based on typical settings for inertial versions of swarm dynamics algorithms. The distribution type is uniform, which is suitable for exploring a broader range of solutions.
# The selector used in this implementation is 'greedy', aiming to balance between exploration and exploitation by prioritizing the most promising individuals in the population. This approach is justified as it is effective for nonconvex optimization problems like the Rastrigin function where efficiency in finding good solutions is crucial.