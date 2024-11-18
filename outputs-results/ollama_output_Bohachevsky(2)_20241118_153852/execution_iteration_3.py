# Name: natural Computing Algorithm
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)
prob = fun.get_formatted_problem()

heur = [
    (  
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'constriction'
    ),
    (
        'random_sample', 
        {},
        'greedy'
    ),
    (
        'swarm_dynamic', 
        {
            'factor': 1.0,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'inertial'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The designed metaheuristic uses four search operators: 'local_random_walk', 'spiral_dynamic', 'gravitational_search' and 'random_sample'. 
# These operators are used with different selection methods like 'greedy', 'metropolis', 'inertial' and 'constriction'. 
# The parameters of each operator are tuned to optimize the problem. 
# The 'swarm_dynamic' operator uses a bigger selector, as it has more space to cover the search space.