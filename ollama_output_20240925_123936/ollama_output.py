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
    'genetic_mutation',  
    {  
        'scale': 1.0,
        'elite_rate': 0.1,
        'mutation_rate': 0.25,
        'distribution': 'gaussian',
       },
    'greedy'  
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

## Explanation of metaheuristic creation code:


