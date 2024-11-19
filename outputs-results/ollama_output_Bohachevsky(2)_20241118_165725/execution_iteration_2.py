# Name: Hybrid Metaheuristic with Random Sample and Local Random Walk

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)  # The selected problem is Bohachevsky with dimension 2.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',  # Using the random sample operator for initial exploration
        {},
        'greedy'
    ),
    (
        'local_random_walk',  # Using the local random walk operator for fine-tuning
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This hybrid metaheuristic combines two operators: 
# 1. `random_sample` for initial exploration to cover a wide range of the search space.
# 2. `local_random_walk` for fine-tuning near the optimal solution, leveraging local randomness based on a predefined probability.

# The use of `random_sample` ensures that the algorithm starts with diverse solutions, increasing the likelihood of finding the global optimum.

# The `local_random_walk` operator introduces controlled randomness during refinement, allowing the algorithm to navigate through local minima and find more precise optima.

# Together, these operators provide a balance between exploration and exploitation, making the hybrid metaheuristic robust for the Bohachevsky function in two dimensions.