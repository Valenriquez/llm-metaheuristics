# Name: HyperOptimist
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.funktor({self.dimensions}) 
prob = fun.get_formatted_problem()

heur = [
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
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
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# This metaheuristic combines genetic mutation and spiral dynamic operators to optimize the funky problem. 
# The genetic mutation operator is used with a scale of 1.0 and a mutation rate of 0.25, while the spiral dynamic 
# operator is used with a radius of 0.9 and an angle of 22.5 degrees. 
# Both the greedy and metropolis selection strategies are employed to select the next population members.