# Name: Randomized Search with Spiral Dynamic

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        'random_sample',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic uses the Spiral Dynamic operator to explore the solution space in a spiral manner, guided by its parameters (radius, angle, sigma). 
# It also incorporates random sampling to ensure diverse exploration of the search space. This hybrid approach aims to balance exploitation and exploration for better convergence.