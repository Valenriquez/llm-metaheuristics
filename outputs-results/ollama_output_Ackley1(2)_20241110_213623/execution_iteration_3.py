# Name: AckleyMeta
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
    (  # Search operator for Ackley function
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
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

heur2 = [
    (  # Search operator for Ackley function
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# Short explanation and justification:
# 
# The Ackley function is a multi-modal optimization problem. We will use three different metaheuristic algorithms to optimize this problem.
# 
# The first algorithm, named 'AckleyMeta', uses the 'gravitational_search' operator to find a solution to the Ackley function.
# This algorithm is suitable for large-scale optimization problems because it has a relatively low computational cost and can handle complex structures of fitness landscapes.
# However, it may not be as efficient on small-scale problems where other operators might perform better.
# 
# The second algorithm, also named 'AckleyMeta', uses the 'genetic_mutation' operator to find a solution to the Ackley function.
# This algorithm is suitable for small-scale optimization problems because it has a relatively high computational cost but can handle complex structures of fitness landscapes and provides good results on well-designed problem instances.
# However, it may not be as efficient on large-scale optimization problems where other operators might perform better.
# 
# The third algorithm uses the 'spiral_dynamic' operator to find a solution to the Ackley function.
# This algorithm is suitable for optimization problems with many local optima and is particularly useful when the number of dimensions is small, as it can be very efficient on well-designed problem instances.
# However, its performance may degrade for larger numbers of dimensions.