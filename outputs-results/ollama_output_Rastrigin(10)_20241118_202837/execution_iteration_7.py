# Name: Hybrid Metaheuristic with Random Walk and Spiral Dynamic
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
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    ),
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
# This metaheuristic combines the strengths of three different optimization techniques: 
# Swarm Dynamic, Spiral Dynamic, and Local Random Walk. 

# The Swarm Dynamic operator is used to perform a global search, helping to avoid local minima 
# by moving particles across the entire search space.

# The Spiral Dynamic operator guides particles towards the current best solution in a spiral pattern, 
# which can help fine-tune the optimization process around the best-known solutions.

# The Local Random Walk operator introduces randomness into the search process, allowing particles to explore 
# areas of the search space that they might otherwise overlook. This helps prevent premature convergence 
# and encourages diversification.

# The 'all' selector is used for Swarm Dynamic and Spiral Dynamic to ensure coverage over the entire search space.
# The 'probabilistic' selector is used for Local Random Walk, allowing it to occasionally take random steps that increase exploration.