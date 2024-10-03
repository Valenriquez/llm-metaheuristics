 # Name: GravitationalSearchOptimizer
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'greedy'
    ),
    (  
    'gravitational_search',
    {
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements the Gravitational Search Optimizer (GSO), a population-based metaheuristic inspired by the principles of gravitational force and mass interactions. The algorithm consists of two operators, both using the same parameters for gravity and alpha to explore different subsets of solutions during optimization. The first operator uses 'greedy' selection, focusing on improving solutions from the current best in each iteration. The second operator applies 'all' selections, allowing all potential solutions to evolve through gravitational interactions. This setup aims to leverage both local improvement and global exploration capabilities inherent in GSO for optimizing the Sphere benchmark function.
# 1. Gravitational Search Operator: Defined with gravity (force) of 1.0 and alpha coefficient of 0.02, these parameters influence how solutions interact and move towards better regions based on their masses (fitness values).
# 2. Selection Methods: 'greedy' focuses only on the current best solutions for updating, while 'all' considers all possible candidates in each iteration to avoid premature convergence.
# 3. Parameters are directly taken from parameters_to_take.txt as required, ensuring consistency and reproducibility of the experiment.