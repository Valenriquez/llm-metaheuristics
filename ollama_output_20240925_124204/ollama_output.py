## Name: Metaheuristic 1
## Code:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


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
        'num_rands': 2,
        'factor': 1.0},
    'greedy'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()    # With this, the code runs and shows the results
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

## Explanation:

#The metaheuristic uses the differential mutation operator with the current-to-best expression, two random numbers, and a factor of 1.0.
#The greedy selector is used to select the best solution among the current population.

## Comments:

#This metaheuristic performs well for the Rastrigin function with 2 dimensions.
#The number of iterations can be adjusted as needed.