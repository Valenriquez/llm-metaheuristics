# Name: Ackley1 Metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(3)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.9,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample', 
        {
        }, 
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Ackley1 function is a global minimization problem. It was designed to have a very long minimum path in the search space 
# with many local minima, which makes it challenging for traditional metaheuristics. The goal of this metaheuristic algorithm is 
# to use two different operators: local_random_walk and random_sample. Local_random_walk is used to find local minima and then 
# we use random_sample to randomly explore the remaining search space in order to get a better solution, avoiding getting stuck 
# in local minima. This approach should improve the chances of getting smaller fitness solutions.