# Name: Hybrid Metaheuristic with Spiral Dynamic and Local Random Walk
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
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The hybrid metaheuristic combines the Spiral Dynamic operator with a Local Random Walk. 
# The Spiral Dynamic helps in exploring the solution space efficiently by moving along spirals, which is particularly effective for multi-dimensional problems like Rastrigin's function.
# The Local Random Walk further refines the search around promising areas, improving convergence and exploitation capabilities.
# Together, these operators provide a balance between exploration and exploitation, making the metaheuristic well-suited for tackling complex optimization problems.