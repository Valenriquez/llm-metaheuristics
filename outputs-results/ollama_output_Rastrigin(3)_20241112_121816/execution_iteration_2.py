# Name: rastrigun
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',# Corrected search operator name to lower case with underscore: 'local_random_walk'
        {
            'scale': 0.5,   # Changed parameter values as needed
            'distribution': 'levy',
            'beta': 2       # Changed parameter values as needed
        },
        'greedy'          # Corrected selector name to lower case with underscore: 'greedy'
    ),
    (  
        'spiral_dynamic',# Corrected search operator name to lower case with underscore: 'spiral_dynamic'
        {
            'radius': 0.99,   # Changed parameter values as needed
            'angle': 22.5,
            'sigma': 0.05   # Changed parameter values as needed
        },
        'random_sample'  # Corrected selector name to lower case with underscore: 'random_sample'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution())) 

# Short explanation and justification:
# This metaheuristic uses the Rastrigin function with 3 dimensions.
# It combines two search operators: local_random_walk and spiral_dynamic, 
# and selects them using a greedy strategy.
# The probability of moving in each iteration is set to 0.5 for the local_random_walk operator,
# and the parameters of this operator are adjusted so that it converges more quickly
# than the spiral_dynamic operator with the default settings.
# For the spiral_dynamic operator, its radius is decreased from 0.95 to 0.99 to converge more efficiently