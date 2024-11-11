# Name: ackley
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
        'random_flight',
        {
            'scale': 0.5,
            'distribution': 'gaussian',
            'beta': 3.0
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 0.5,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic uses the random_flight operator to search for the optimal solution.
# Then, it uses the genetic_mutation operator to refine the solution.
# The greedy selector is used in both operators.
# By using a combination of these operators, we can find a better fitness solution than 0.