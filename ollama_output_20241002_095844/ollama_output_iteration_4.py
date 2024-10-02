 # Name: CustomSwarmDynamicHeuristic
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
# The metaheuristic named CustomSwarmDynamicHeuristic is designed to optimize the Rastrigin function with two dimensions using a swarm dynamic algorithm. The algorithm has been configured with specific parameters including factor, self_conf, swarm_conf, version, and distribution as per the requirements from the parameters_to_take.txt file. The selector used in this implementation is Metropolis which is appropriate for global optimization problems where local search methods are beneficial to avoid premature convergence.
# The choice of 'swarm_dynamic' operator is justified based on its ability to dynamically adjust swarm behavior, allowing for exploration and exploitation trade-offs that can help in navigating complex landscapes such as the Rastrigin function with multiple local minima. The parameters set include a factor of 0.7 which controls the influence of cognitive and social components, self_conf sets the confidence level for individual particles, swarm_conf adjusts the group's collective intelligence, version refers to the type of inertia weight adjustment (inertial in this case), and distribution is defined as uniform to randomly perturb particle positions within a certain range.
# The selector 'metropolis' is selected because it allows the algorithm to occasionally accept worse solutions in hopes of escaping local minima, which aligns with the need for global optimization where multiple local optima can exist such as in the Rastrigin function.