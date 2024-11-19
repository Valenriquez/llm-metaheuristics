# Name: Multi-Operator Adaptive Search (MOAS)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        # Search operator 2: Random Sample
        'random_sample',
        {},
        'all'
    ),
    (
        # Search operator 3: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# MOAS combines three different search operators to tackle the Rastrigin function. The Local Random Walk operator helps in exploring the neighborhood of the current solution, which is useful for fine-tuning the solution. The Random Sample operator ensures that the algorithm does not get stuck in local optima by occasionally proposing completely new solutions. The Spiral Dynamic operator provides a structured exploration pattern, guiding the search towards better regions in the solution space. This combination allows the algorithm to balance exploitation and exploration effectively, making it well-suited for multi-dimensional problems like Rastrigin.