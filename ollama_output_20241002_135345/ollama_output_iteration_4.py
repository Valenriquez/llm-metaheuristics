 # Name: CustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operator using parameters from parameters_to_take.txt
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

# Create the metaheuristic instance with the defined search operator and problem
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is named CustomMetaheuristic, aiming to solve the optimization problem using a swarm dynamic search operator with parameters set as specified in parameters_to_take.txt. The swarm_dynamic operator is selected using the metropolis selector, which fits well for probabilistic selection based on acceptance criteria. This approach aims to balance exploration and exploitation effectively by adapting the factor and configuration settings appropriately according to the recommended values from the parameter file. The use of uniform distribution ensures a diverse initial population that can be adjusted during iterations to better explore the solution space. The verbose setting is enabled to track the optimization progress, providing insights into how the algorithm evolves over time.