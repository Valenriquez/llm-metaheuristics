 # Name: PSOwithAdaptiveParameters
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'PSO',
    {
        'chi': 0.7,
        'r_1': lambda: random.random(),
        'r_2': lambda: random.random(),
        'self_conf': 1.5,
        'swarm_conf': 1.5
    },
    'PSO'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The PSOwithAdaptiveParameters metaheuristic is designed to implement a Particle Swarm Optimization (PSO) algorithm with adaptive parameters. 
# The algorithm uses the default PSO settings where 'chi' represents the cognitive component, 'r_1' and 'r_2' are random factors, 
# 'self_conf' adjusts the influence of an agent on itself, and 'swarm_conf' affects how agents learn from the global best position. 
# These parameters are set with specific values or functions to dynamically adapt during the optimization process, which is suitable for exploring a nonconvex problem like Rastrigin function. 
# The selector_name is also set to PSO, ensuring that both genetic crossover and mutation are applied as part of the PSO algorithm's operation.