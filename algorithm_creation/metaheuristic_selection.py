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



# A Metaheuristic is an iterative procedure. It renders an optimal solution for a given optimisation problem 
# with an objective function f(x), using using a finite sequence of simple heuristics (SHs). 
# These SHs are applied iteratively until a stopping condition is met. 
#  An inherent property of metaheuristics is the cardinality, which is defined as the number of search operators 
# implemented in it, while disregarding its initialiser and finaliser. 


# A given metaheuristic contains: 
# - One perturbator, which is the operators that modify the solution
add__operator__ = ['local_random_walk', 'random_search', 'random_sample', 'random_flight', 'differential_mutation',
           'firefly_dynamic', 'swarm_dynamic', 'gravitational_search', 'central_force_dynamic', 'spiral_dynamic',
           'genetic_mutation', 'genetic_crossover']

# - a selector
# - parameters for the perturbator
# and the population selector, such as: 

add__selector__ = ['greedy', 'probabilistic', 'metropolis', 'all', 'none']

print("----------Metaheuristic creation-------------------")
fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()
heur = [( # Search operator 1
    'add__operator__ ',  # Perturbator
    {  # Parameters
        'expression': 'current-to-best',
        'num_rands': 2,
        'factor': 1.0},
    'add__selector__'  # Selector
), (  # Search operator 2
    'add__operator__',  # Perturbator
    {  # Parameters
        'crossover_rate': 0.2,
        'version': 'binomial'
    },
    'add__selector__'  # Selector
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()    # With this, the code runs and shows the results
#print('x_best = {}, f_best = {}'.format(*met.get_solution()))
