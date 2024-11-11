# Name: swarm_dynamic_1
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# We designed a metaheuristic algorithm called "swarm_dynamic_1" to find the global minimum of the Ackley function with dimension 2.
# The search operator is set to "random_flight", which is known for its good performance in multi-modal optimization problems.
# The local_random_walk operator is used as the secondary search operator to help navigate through the search space and avoid getting stuck in local optima.
# We use a greedy selector to choose the next search operator at each iteration, and a metropolis selector to decide whether to accept or reject the new solution.
# Our algorithm runs for 100 iterations and prints out the best solution found.