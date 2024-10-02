 # Name: CustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Using greedy selector and probabilistic operator for exploration
operator = {
    'type': 'swarm_dynamic',
    'parameters': {
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'gaussian'
    }
}
selector = 'probabilistic'

heur = [(
    '[operator_name]',
    {
        'type': operator['type'],
        'parameters': operator['parameters']
    },
    selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named CustomMetaheuristic is designed to solve optimization problems using a combination of a probabilistic operator (swarm_dynamic) and a selective strategy (probabilistic). This approach is chosen based on the recommendation for ill-structured global optimization problems where nonconvex, nondifferentiable, and possibly discontinuous objective functions are common.
# The swarm_dynamic operator is configured with specific parameters: factor set to 0.7, self_conf to 2.54, swarm_conf to 2.56, version set to inertial, and distribution type to gaussian. These settings aim to balance exploration and exploitation during the search process.
# The probabilistic selector ensures that decisions are made based on probabilities derived from the current state of the search space, which is beneficial for navigating complex landscapes where a random component can be advantageous.