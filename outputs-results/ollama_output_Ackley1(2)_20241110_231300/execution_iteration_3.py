# Name: ackley_natural_computing
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
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
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'probabilistic'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The proposed metaheuristic combines the advantages of different operators. 
# The gravitational search operator is used to initialize the solution, providing a strong global search capability.
# The spiral dynamic operator is employed for local optimization, allowing the algorithm to converge to local minima.
# The random flight operator is used to explore the solution space probabilistically, ensuring a good diversification of the solutions.
# By combining these operators, the metaheuristic can efficiently navigate the Ackley function's landscape and find a smaller solution with a fitness value less than 0.