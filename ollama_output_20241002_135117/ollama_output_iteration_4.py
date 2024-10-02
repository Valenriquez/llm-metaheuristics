 # Name: CustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operator with parameters from parameters_to_take.txt
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

# Create and run the metaheuristic with the defined search operator
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is named CustomMetaheuristic as it represents a customized approach for optimization using the swarm_dynamic operator, which is designed based on the parameters specified in the parameters_to_take.txt file. The Rastrigin function (2-dimensional) is chosen as the benchmark problem because it is commonly used to test the performance of metaheuristics due to its multi-modal and non-convex nature.
# The swarm_dynamic operator is configured with specific parameters: factor set to 0.7, self_conf set to 2.54, swarm_conf set to 2.56, version as inertial, and distribution as uniform. These settings are chosen to balance exploration and exploitation in the search space according to typical recommendations for such operators.
# The selector 'metropolis' is used here because it fits well with the probabilistic nature of the swarm_dynamic operator, enhancing the quality of solutions through a probabilistic selection criteria during iterations. This approach allows for a dynamic adjustment of the swarm behavior, potentially leading to better convergence properties compared to simpler greedy or random search methods.
# The metaheuristic runs for 100 iterations as specified, and results are printed showing the best found solution x_best and its corresponding function value f_best after optimization is completed.