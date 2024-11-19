# Name: Hybrid Metaheuristic with Local Random Walk and Spiral Dynamic

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
    (
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
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named "Hybrid Metaheuristic with Local Random Walk and Spiral Dynamic" combines the strengths of two well-known algorithms:
# 1. **Local Random Walk**: This operator helps in exploring the neighborhood of the current solution, which is beneficial for fine-tuning and avoiding getting stuck in local minima.
# 2. **Spiral Dynamic**: This operator allows for a global exploration strategy by following a spiral path around the search space, ensuring that the algorithm can escape from local minima more effectively.

# The hybrid approach balances both local refinement and global exploration, making it suitable for complex optimization problems like the Bohachevsky function.