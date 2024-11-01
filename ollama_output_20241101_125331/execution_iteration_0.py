 # Name: Adaptive Metropolis-Greedy Algorithm
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'local_random_walk',
        { 
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (  
        'greedy',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The Adaptive Metropolis-Greedy Algorithm is designed to leverage both local search capabilities and global exploration through a combination of operators. 
# The algorithm starts with the 'local_random_walk' operator, which performs random walks influenced by Gaussian distribution for better convergence near the current best solution. 
# This is augmented with the 'greedy' selector that ensures some exploitation in every iteration to refine the search direction and avoid premature convergence.
# By using a 'metropolis' selector for the local_random_walk operator, we ensure a balance between exploration (via random walks) and exploitation (guided by the current best solution), which is crucial for escaping local minima and finding better solutions overall. 
# The use of Gaussian distribution in the local search ensures that the perturbations are significant enough to explore new regions but not too large, maintaining stability around the optimal values found so far.
# This approach aims to combine the adaptive nature of Metropolis-Hastings with the greedy selection method for effective optimization across multiple iterations.