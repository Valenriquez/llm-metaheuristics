# Name: RastriginMeta
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
    (  
        'swarm_dynamic', # Search operator 1 
        {
            'factor': 0.5,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform' # New distribution
        },
        'greedy' # Changed to greedy
    ),
    (
        'spiral_dynamic', # Search operator 2 
        {
            'radius': 0.95,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis' # Changed to metropolis
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The problem was too easy. We need a better distribution to increase the exploration of the search space.
# By setting 'factor': 0.5, we allow the swarm to move further away from its current position, which helps to escape local minima.
# Additionally, by changing the distribution to 'uniform', we ensure that each dimension is explored equally, rather than being biased towards certain areas.
# These modifications should lead to better performance and a smaller fitness solution.