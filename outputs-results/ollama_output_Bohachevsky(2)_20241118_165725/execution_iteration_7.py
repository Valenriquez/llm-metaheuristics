# Name: Enhanced Randomized Metaheuristic (ERMH)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2) # This is the selected problem, the problem may vary depending on the case.
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
        'random_sample',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Enhanced Randomized Metaheuristic (ERMH) combines the local_random_walk operator with a random_sample operator. 
# The local_random_walk operator allows for small, controlled steps around the current solution, helping to refine the search.
# The random_sample operator ensures that the search space is thoroughly explored by randomly sampling new solutions.
# By combining these two operators, the algorithm can efficiently explore both fine-grained details and broad areas of the search space,
# potentially leading to better convergence towards the optimal solution.