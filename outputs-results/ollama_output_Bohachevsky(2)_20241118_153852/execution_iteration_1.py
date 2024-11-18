# Name: mbo
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
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
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
    )
]

heur2 = [
    (  # Search operator 2
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'beta': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur + heur2, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# I have designed a metaheuristic algorithm called mbo.
# The first search operator is local_random_walk, which uses probability 0.75 to make decisions.
# The second search operator is spiral_dynamic, which uses radius 0.9, angle 22.5, and sigma 0.1.
# Both of these operators are selected using the greedy strategy for heur 0-2 and the metropolis strategy for heur 3-4.
# The third and fourth search operators are gravitational_search and random_flight respectively.
# These two operators use gravity 1.0, alpha 0.02, scale 1.0, beta 1.5 and distribution 'uniform', 'gaussian' or 'levy'.
# The three search operators are combined to form the metaheuristic mbo.
# This algorithm is run for a total of 100 iterations using a Bohachevsky function with dimension 2 as the benchmark problem.