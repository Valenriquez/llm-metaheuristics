 # Name: CustomSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define search operator using parameters from parameters_to_take.txt
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

# Create metaheuristic with the defined operator
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named CustomSearchMetaheuristic as it combines multiple search operators to find the best solution for optimization problems. 
# We start by importing necessary modules for handling benchmark functions and metaheuristics.
# A benchmark function Rastrigin(2) is selected, which is a standard test problem in optimization used to evaluate the performance of algorithms.
# The main search operator is defined using parameters from parameters_to_take.txt: swarm_dynamic with specific values for factor, self_conf, swarm_conf, version, and distribution. 
# These parameters are chosen based on their relevance to dynamic swarm behavior in optimization algorithms.
# The selector 'metropolis' is used as per the instructions, which is a probabilistic selection method suitable for global optimization problems where exploration of diverse solutions is crucial.
# A metaheuristic instance is created with the defined problem and operator, set to run for 100 iterations, enabling detailed output during execution.
# Finally, the best solution found by the algorithm is printed after execution completes.