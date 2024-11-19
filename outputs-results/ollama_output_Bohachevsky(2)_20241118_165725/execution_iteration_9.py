# Name: Adaptive Randomized Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'random_sample',
        {},
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named "Adaptive Randomized Metaheuristic" uses a combination of the local random walk operator and the random sample operator.
# The local random walk operator is used to make small, controlled steps around the current solution with a certain probability. This helps in exploring the neighborhood of the current solution.
# The random sample operator allows for larger leaps across the search space without any control, which helps in escaping local minima.
# Both operators are combined in an adaptive manner, where each iteration chooses between them based on the previous results to achieve a balance between exploration and exploitation.