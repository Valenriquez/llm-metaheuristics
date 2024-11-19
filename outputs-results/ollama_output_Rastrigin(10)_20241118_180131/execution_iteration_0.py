# Name: Randomized Search with Spiral Dynamics
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'all'),
    ('spiral_dynamic',
     {
         'radius': 0.9,
         'angle': 22.5,
         'sigma': 0.1
     },
     'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is a combination of random sampling and spiral dynamics. 
# Random sampling helps to explore the solution space without getting stuck in local minima. 
# Spiral dynamics guides the search towards promising regions using a logarithmic spiral pattern, 
# which can be more efficient than simple random walk for some problem landscapes.
#
# By combining these two approaches, we hope to leverage the strengths of both: exploration and exploitation.
# The use of 'all' selector ensures that every operator gets a chance to contribute to each iteration.