# Name: Ackley3
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
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

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# Short explanation and justification:
# The Ackley3 metaheuristic combines the strengths of gravitational search and local random walk to find the global optimum. 
# Gravitational search is used as the primary search operator, which provides a good initial solution.
# Local random walk is used as the secondary search operator, which helps to explore the surrounding area and avoid local optima.

# The selection of parameters for the operators is based on the characteristics of the Ackley1 function. 
# The gravity in gravitational search should be high enough to attract solutions quickly, but not so high that it gets stuck in local optima.
# The probability in local random walk should be low enough to avoid unnecessary movements, but not so low that it fails to explore the surrounding area.

# The use of greedy as the selector for gravitational search ensures that the best solution is chosen at each iteration. 
# The use of metropolis as the selector for local random walk provides a balance between exploring and exploiting.