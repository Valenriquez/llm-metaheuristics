# Name: ackley_metaheuristic
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
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# In this metaheuristic we used two operators that can provide good solutions for the Ackley function. 
# The Gravitational Search operator can be used when we need to find global minima and the Random Flight operator is useful when we want local optima.

# To further improve the performance of our metaheuristic, it would be beneficial to explore different combinations of operators or consider adding other operators.
# In particular, combining gravitational search with random flight could lead to improved solutions.