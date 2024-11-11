# Name: Ackley_With_Swarm
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
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'random_sample'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# I have designed a metaheuristic algorithm called Ackley_With_Swarm. 
# The algorithm uses two search operators: Gravitational_Search and Local_Random_Walk.
# Both of these operators use the random_sample selector to select solutions for each iteration. This helps to avoid getting stuck in local optima.
# In addition, the Gravitational_Search operator is designed to pull towards the center of the Ackley function, which encourages the algorithm to explore different parts of the search space and converge on a global minimum.