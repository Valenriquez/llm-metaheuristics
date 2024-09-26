 # Name: Custom Metaheuristic 1
# Code:
import sys
import os

sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)
prob = fun.get_formatted_problem()

heur = [ # Search operator 
( # Search operator 2 (if used)
    'differential_mutation',
    {'mutation_rate': 0.5, 'crossover_rate': 0.7},
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# I chose 'local_random_walk' with a step size of 0.1 and greedy selector for the first operator as it is effective in exploring local areas around the current solution. The second operator is 'differential_mutation' with mutation rate set to 0.5 and crossover rate to 0.7, using probabilistic selection to balance exploration and exploitation. These choices are based on typical settings that promote both diversity and convergence in optimization problems.