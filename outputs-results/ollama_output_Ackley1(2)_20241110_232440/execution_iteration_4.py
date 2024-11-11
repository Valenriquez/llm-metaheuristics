# Name: ackley_one
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
    (
        'genetic_mutation',
        {
            'scale': 0.5,
            'elite_rate': 0.05,
            'mutation_rate': 0.15
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 23,
            'sigma': 0.1
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines genetic mutation and spiral dynamic operators.
# Genetic mutation is used to introduce random changes into the solution space,
# while spiral dynamic operator explores the neighborhood of promising solutions.