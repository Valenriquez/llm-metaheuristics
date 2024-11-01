 # Name: Adaptive Random Walk Metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'local_random_walk',
    { 
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'uniform' or 'gaussian' or 'levy'
    },
    'probabilistic'
    ),
    (  
    'random_sample',
    {
        'selector': 'metropolis'
    },
    'metropolis'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is an Adaptive Random Walk Metaheuristic, designed to balance exploration and exploitation through probabilistic selection based on the probability parameter. 
# The first operator 'local_random_walk' employs a random walk with a specified probability of movement (0.75), scale (1.0), and distribution type ('uniform', 'gaussian', or 'levy'). 
# This allows for both local exploration and global search capabilities depending on the setting.
# The second operator 'random_sample' uses a probabilistic selector named 'metropolis'. This selector is appropriate as it matches the requirement for selecting candidates with a probability-based mechanism, which aligns well with the adaptive random walk approach.
# Both operators are integrated into the metaheuristic framework to ensure a diverse search strategy that can adapt to different problem characteristics through parameter tuning. 
# The combination of local exploration and global sampling aims to optimize the Rastrigin function effectively by adjusting probability and distribution settings based on the parameters_to_take.txt specifications.