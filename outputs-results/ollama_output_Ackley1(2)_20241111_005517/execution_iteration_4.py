# Name: ackley_metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 22.5,
            'sigma': 0.05
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# The Ackley function is a non-convex optimization problem. We should be able to find a better solution by using different search operators and 
# selection methods.
# The 'greedy' selector chooses the search operator with the best fitness at each step, which may not always lead to the global minimum. Using the 
# 'probabilistic' selector instead can help escape local minima more effectively.
# The 'gravitational_search' operator is a metaheuristic that uses a particle swarm optimization approach to find the optimal solution.
# The 'spiral_dynamic' operator uses a spiral search pattern to explore the search space, which can be effective for non-convex problems.