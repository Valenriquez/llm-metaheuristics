# Name: Hybrid Metaheuristic with Random Flight and Local Random Walk
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
        'random_flight',
        {
            'scale': 0.5,
            'distribution': 'uniform',
            'beta': 1.5
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines Random Flight and Local Random Walk operators to explore the search space effectively.
# Random Flight helps in covering a large area quickly, while Local Random Walk refines the solution locally.
# The 'all' selector for Random Flight ensures it is used throughout the entire search process.
# The 'greedy' selector for Local Random Walk helps in rapidly converging to high-quality solutions.