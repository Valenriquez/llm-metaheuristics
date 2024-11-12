# Name: spiral_dynamic_meta
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(5)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
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

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# I designed a metaheuristic algorithm that combines the random flight and local random walk operators.
# The random flight operator is used as the primary search operator, while the local random walk operator is used to 
# explore the neighborhood of the current solution. The greedy selector is used to select the best solution from the 
# population at each iteration. I chose these operators because they are both popular in nature-inspired optimization 
# algorithms and can be effective in converging to good solutions. 

# Additionally, the Ackley function is a difficult optimization problem, so it requires careful tuning of the parameters. 
# In this case, I used a scale factor of 1.0 for the random flight operator and a probability of 0.75 for the local 
# random walk operator. I also set the distribution to uniform for the local random walk operator. 

# The main goal of this metaheuristic is to get a smaller fitness solution than the actual fitness, which is 0 in this case.
# This requires careful tuning of the parameters and may require more iterations to converge to a good solution.