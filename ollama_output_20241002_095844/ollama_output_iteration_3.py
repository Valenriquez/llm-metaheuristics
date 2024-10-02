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
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is named CustomMetaheuristic as it represents a custom approach to optimization using the swarm_dynamic operator with specific parameters for exploration in the search space.
# We selected the Rastrigin function due to its nonconvex nature, which aligns well with the purpose of using random search algorithms designed for ill-structured global optimization problems.
# The swarm_dynamic operator is configured with a factor of 0.7 and self_conf set to 2.54, while swarm_conf is adjusted to 2.56 to balance exploration and exploitation. The version is chosen as inertial, and the distribution type is uniform to encourage diversity in the search population.
# The probabilistic selector ensures that the algorithm relies on random elements to guide its search process, which is suitable for problems with multiple local optima where a purely deterministic approach might get stuck in suboptimal solutions.
# This setup aims to leverage the adaptive properties of swarm dynamics and probabilistic selection to efficiently navigate through complex landscapes in search of global optimality.