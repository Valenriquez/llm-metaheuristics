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
    (  # Search operator 1
        'random_flight',  # Changed to random_flight
        {
            'scale': 0.5,  # Increased scale for better exploration
            'distribution': 'levy',  # Maintained distribution
            'beta': 2  # Decreased beta for better convergence
        },
        'greedy'  # Changed to greedy
    ),
    (
        'swarm_dynamic',  # Changed to swarm_dynamic
        {
            'factor': 0.7,  # Increased factor for better exploration
            'self_conf': 2.56,
            'swarm_conf': 2.65,
            'version': 'inertial',
            'distribution': 'uniform'  # Maintained distribution
        },
        'metropolis'  # Changed to metropolis
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The problem is still challenging. We need a better distribution to increase the exploration of the search space.
# By setting 'factor': 0.7, we allow the swarm to move further away from its current position, which helps to escape local minima.
# Additionally, by increasing the scale for 'random_flight' and changing the distribution to 'uniform', we ensure that each dimension is explored equally, rather than being biased towards certain areas.
# These modifications should lead to better performance and a smaller fitness solution.