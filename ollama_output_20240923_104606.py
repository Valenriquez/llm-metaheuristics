## Name: Differential Mutation with Greedy Selector

## Code:
 
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

heur = [( # Search operator 1
    'differential_mutation',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 1,
        'factor': 1.0},
    'greedy'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
 

## Explanation of the code:

#This code implements the Differential Mutation metaheuristic with the Greedy selector. It uses the default parameters for the Differential Mutation operator and the Greedy selector.