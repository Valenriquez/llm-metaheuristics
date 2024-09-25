## Name: Metaheuristic with Differential Mutation and Greedy Selector
## Code:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import pkgutil

sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')

import benchmark_func as bf
import population as pp
import operators as op
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [(
    'differential_mutation',
    {'expression': 'current-to-best', 'num_rands': 1, 'factor': 1.0},
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

## Explanation of the code:
#This code uses the differential mutation operator with the current-to-best expression, a num_rands of 1, and a factor of 1.0. The greedy selector is used to select the best solution in each iteration.