# Name: Hybrid Metaheuristic for Rastrigin Function
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
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'gaussian'}, 'greedy'),
    ('random_sample', {}, 'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of local search (local_random_walk) and global exploration (random_sample).
# The local_random_walk operator helps in fine-tuning the solution by making small, controlled moves. 
# The random_sample operator ensures that the algorithm explores a broader range of the solution space.
# Together, these operators balance exploitation and exploration, leading to better performance on the Rastrigin function.