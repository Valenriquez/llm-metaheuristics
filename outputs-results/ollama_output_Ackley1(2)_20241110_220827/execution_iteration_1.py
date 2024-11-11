# Name: ackley_one
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_flight',
        {
            'scale': 0.5,
            'distribution': 'levy',
            'beta': 2.5
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 0.3,
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# I have designed a metaheuristic algorithm that combines the random flight operator with the local random walk operator.
# The random flight operator has a smaller scale, which allows for more exploration in the search space.
# The local random walk operator has a higher probability of moving in the current direction, which allows for more exploitation of good solutions.
# This combination allows the metaheuristic to find better solutions than using only one operator.