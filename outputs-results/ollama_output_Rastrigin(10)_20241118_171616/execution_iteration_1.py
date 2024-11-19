# Name: Hybrid Metaheuristic with Spiral and Local Random Walk

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # The selected problem is Rastrigin with 10 dimensions.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all' # Use a large selector as the problem is in 10 dimensions
    ),
    (
        'local_random_walk', # Search operator 2: Local Random Walk
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'all' # Use a large selector as the problem is in 10 dimensions
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The hybrid metaheuristic combines the Spiral Dynamic operator with a Local Random Walk to effectively explore and exploit the search space of the Rastrigin function. 
# Spiral Dynamic helps in covering large regions, while the Local Random Walk refines the solution by making fine adjustments. This combination is particularly suitable for high-dimensional problems like Rastrigin.