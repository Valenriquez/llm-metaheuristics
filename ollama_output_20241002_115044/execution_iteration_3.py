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
# The chosen metaheuristic is named CustomMetaheuristic as it combines elements from various well-known heuristics to create a unique approach for optimization problems.
# We selected the Rastrigin function, which is suitable for testing due to its many local minima, making it an ideal candidate for global optimization algorithms.
# The search operator 'swarm_dynamic' is chosen based on its ability to explore and exploit solutions dynamically during the optimization process using parameters that control the behavior of individual particles within a swarm (factor, self_conf, swarm_conf) and how they move through the search space (version). 
# Distribution type is set to uniform for initial exploration. The selector 'greedy' prioritizes immediate improvement over broader exploration, which can be beneficial in scenarios where quick convergence is more critical than extensive search. This setup aims to balance between exploration and exploitation effectively.