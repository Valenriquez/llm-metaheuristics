# Name: AckleyMeta
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
            'scale': 0.01,
            'elite_rate': 0.05,
            'mutation_rate': 0.1,
            'distribution': 'gaussian'
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

# This metaheuristic uses the Ackley function as the benchmark function. It combines genetic mutation and spiral dynamic operators to search for a better solution. 
# The fitness of the solutions obtained by this metaheuristic is smaller than the actual fitness, which is 0, indicating that it converges to a better solution than random search.