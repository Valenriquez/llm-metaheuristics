# Name: Hybrid Metaheuristic with Random Sample and Local Random Walk

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'all'
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
# The Hybrid Metaheuristic with Random Sample and Local Random Walk combines the benefits of both random exploration 
# (provided by the 'random_sample' operator) and targeted exploitation (using a local random walk approach guided by the 'local_random_walk' operator). 

# The 'random_sample' operator allows for a broad search space exploration, while the 'local_random_walk' operator refines the solution
# around promising regions identified during the sampling phase. This hybrid strategy can help to balance between finding new potential 
# solutions and efficiently improving the current best-known solution.

# By utilizing both operators, the metaheuristic aims to improve its overall performance on complex optimization problems like Rastrigin's function.