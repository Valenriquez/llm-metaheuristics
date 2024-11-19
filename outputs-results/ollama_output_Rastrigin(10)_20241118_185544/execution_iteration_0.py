# Name: Hybrid Metaheuristic with Local Random Walk and Spiral Dynamic
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
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the local_random_walk operator with a spiral_dynamic operator to effectively search both locally around optimal solutions and globally across the solution space. The use of 'greedy' for local_random_walk helps in quickly converging to better solutions, while 'all' selector for spiral_dynamic ensures thorough exploration of the solution space. This hybrid approach is well-suited for complex benchmark functions like Rastrigin, which often require both exploitation and exploration strategies for optimal performance.