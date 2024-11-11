# Name: Ackley Metaheuristic
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
        'genetic_mutation',
        {
            'scale': 0.5,
            'elite_rate': 0.05,
            'mutation_rate': 0.2,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'random'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
#
# The Ackley metaheuristic algorithm is a simple genetic mutation operator combined with a local random walk. 
# This combination allows the algorithm to efficiently explore the solution space and converge to good solutions.
#
# In this implementation, the 'genetic_mutation' operator uses a scale of 0.5 and an elite rate of 0.05, 
# which means that 5% of the population is kept as the best individuals at each iteration. 
# The mutation rate is set to 0.2, which means that 20% of the population mutates to new solutions.
#
# On the other hand, the 'local_random_walk' operator uses a probability of 0.8 and a scale of 1.0, 
# which means that the algorithm walks around the current solution with a probability of 80%. 
# The distribution used is uniform, which means that the algorithm moves in any direction.
#
# The selection strategy used is 'greedy', which means that the algorithm chooses the best individual as the next solution.