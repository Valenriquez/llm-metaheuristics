# Name: Ackley1_Chill
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
    (
        'random_sample',
        {
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
# To get a smaller fitness solution, we use the 'random_sample' operator to randomly select solutions from the population,
# and then use the 'spiral_dynamic' operator to search for better solutions. This combination helps to avoid getting stuck in local optima.
# The 'greedy' selector is used to ensure that the best individual is selected for the next generation, which helps to converge to the global optimum.