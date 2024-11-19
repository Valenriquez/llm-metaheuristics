# Name: Hybrid Metaheuristic with Random Flight and Local Random Walk

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'uniform',
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
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Hybrid Metaheuristic combines two operators: `random_flight` and `local_random_walk`. 
# `random_flight` helps in exploring the solution space widely by taking large steps based on a uniform distribution.
# `local_random_walk` is used for fine-tuning the solution around the current best position, leveraging Gaussian noise.
# The combination of these two operators allows the algorithm to efficiently explore and exploit the search space,
# making it well-suited for solving optimization problems like Rastrigin's function.