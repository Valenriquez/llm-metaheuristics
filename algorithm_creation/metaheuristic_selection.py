import matplotlib.pyplot as plt
import sys
 
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


print("----------Metaheuristic creation-------------------")

fun = ...

prob = fun.get_formatted_problem()
heur = [( # Search operator 1
    'add__operator__',  # Perturbator
    {  # Parameters
    },
    'add__selector__'  # Selector
), ( # Search operator 2
    'add__operator__',  # Perturbator
    {  # Parameters
        'parameters': ""
    },
    'add__selector__'  # Selector
)]


met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()    # With this, the code runs and shows the results
#print('x_best = {}, f_best = {}'.format(*met.get_solution()))
