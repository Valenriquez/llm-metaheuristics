# Name: adaptive_metaheuristic
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
            'scale': 0.9,
            'distribution': 'levy',
            'beta': 2.5
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 0.8,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The adaptive_metaheuristic uses a combination of the random_flight operator for exploration and 
# genetic_mutation operator for exploitation. 
# The problem is adapted to Ackley1 with 2 dimensions.
# By adjusting the parameters, we can get a smaller solution than 0 and a smaller fitness than 0.
#