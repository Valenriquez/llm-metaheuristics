# Name: ackley_mh
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
            'distribution': 'levy',
            'beta': 3.0
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'elite_rate': 0.2,
            'mutation_rate': 0.1,
            'scale': 1.5,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation:
# This metaheuristic uses the random_flight operator to explore the search space and the genetic_mutation operator to refine solutions.
# The random_flight operator uses a Levy flight distribution with a scaling factor of 0.5 and a beta value of 3.0, which provides a good balance between exploration and exploitation.
# The genetic_mutation operator uses an elite rate of 0.2 and a mutation rate of 0.1, which allows for some genetic drift while maintaining the overall diversity of the population.
# The metropolis selector is used to select the next operator based on the current solution's fitness.
# The goal of this metaheuristic is to find a solution that is both better than 0 and has a lower fitness value than 0.