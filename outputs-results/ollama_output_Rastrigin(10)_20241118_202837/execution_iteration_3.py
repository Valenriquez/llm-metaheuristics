# Name: Hybrid Metaheuristic with Random Sampling and Local Walks
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Local Random Walks for fine-tuning
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample', # Search operator 2: Random Sampling for exploration
        {
        },
        'all'  # Use a bigger selector for higher dimensions
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The proposed metaheuristic combines local random walks with random sampling to balance exploration and exploitation. Local random walks are particularly effective for fine-tuning solutions, ensuring that the search does not get stuck in suboptimal regions. Random sampling, which uses a 'greedy' selector, provides a broader search space and helps escape local optima. This hybrid approach leverages the strengths of both strategies to find better solutions more efficiently.