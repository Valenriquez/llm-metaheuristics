# Name: Ackley-3
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
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# Short explanation and justification:
# The Ackley-3 metaheuristic combines the genetic mutation operator with local random walk for exploration and exploitation. 
# It uses a greedy selector to choose the operators at each iteration, allowing the algorithm to adapt to changing problem environments.
#
# In this implementation, we use a fixed scaling factor and elite rate for the genetic mutation operator, while tuning the probability and distribution parameters for the local random walk operator.