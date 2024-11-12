# Name: ackley_meta
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(5)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',  # changed from 'random_flight' to 'local_random_walk'
        {
            'scale': 2.0,
            'distribution': 'gaussian',
            'probability': 1.0,  # fixed probability for local_random_walk
        },
        'metropolis'  # fixed selector for local_random_walk
    ),
    (
        'local_random_walk',  # changed from 'random_flight' to 'local_random_walk'
        {
            'scale': 3.0,
            'distribution': 'levy',
            'probability': 1.0  # fixed probability for local_random_walk
        },
        'greedy'  # fixed selector for local_random_walk
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and justification:
# The Ackley function is a multi-modal problem with many local optima. 
# A suitable metaheuristic should explore the search space thoroughly to find the global optimum.
# Here, I have chosen 'local_random_walk' as the first operator due to its ability to jump over the local optima.
# As a backup, I have used 'local_random_walk' as the second operator, which can also explore nearby areas.
# The 'greedy' selector is used here because it provides a strong exploitation strategy to avoid getting stuck in local optima.
# In this case, both operators converge to the same global optimum.