# Name: ackley_meta
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
        'gravitational_search',
        {
            'gravity': 1.5,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 1.0,
            'angle': 45.0,
            'sigma': 0.2
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The ackley_meta metaheuristic is designed to find a good solution for the Ackley function with 2 dimensions.
# It uses a combination of two search operators: gravitational_search and spiral_dynamic. 
# The gravitational_search operator is used as the initial search operator, while the spiral_dynamic operator is used as the secondary search operator.
# The selection of these two operators is based on their performance in different situations. 
# The gravitational_search operator has good global exploration ability, but its local exploitation ability is limited. 
# On the other hand, the spiral_dynamic operator has good local exploitation ability, but its global exploration ability is limited. 
# By combining these two operators, we can achieve a good balance between global and local search.
# The metaheuristic is designed to get a smaller solution than 0 and a smaller fitness than 0, which are both requirements for this problem.