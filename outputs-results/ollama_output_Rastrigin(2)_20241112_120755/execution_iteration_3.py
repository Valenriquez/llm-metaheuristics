# Name: OptimizedRastriginMeta
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic', 
        {
            'factor': 0.7, # Increased factor for better exploration
            'self_conf': 2.56,
            'swarm_conf': 2.65,
            'version': 'inertial',
            'distribution': 'uniform' # New distribution
        },
        'metropolis' 
    ),
    (
        'spiral_dynamic', # Search operator 2 
        {
            'radius': 0.95, # Increased radius for better exploration
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy' # Changed to greedy
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))


# Short explanation and justification:
# The problem is still challenging. We need a better distribution to increase the exploration of the search space.
# By setting 'factor': 0.7, we allow the swarm to move further away from its current position, which helps to escape local minima.
# Additionally, by increasing the radius for 'spiral_dynamic' and changing the distribution to 'uniform', we ensure that each dimension is explored equally, rather than being biased towards certain areas.
# These modifications should lead to better performance and a smaller fitness solution.