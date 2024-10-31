 # Name: Adaptive Random Walk with Metropolis Selection
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
        'distribution': 'gaussian'
    },
    'metropolis'
    ),
    (  
    'random_sample',
    {
        'selector': 'probabilistic'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic introduces a local random walk operator with parameters for probability and scale, as well as the distribution type (Gaussian in this case). The selector for both operators is set to 'metropolis', which aligns with the probabilistic approach typical of Metropolis algorithms. 
# The use of Gaussian distribution in the local_random_walk operator allows for exploration of the search space based on a normal distribution centered around the current position, promoting diverse yet focused movement within the optimization process. 
# The random_sample operator is configured with 'probabilistic' selector to ensure that sampling decisions are made according to predefined probabilities, which can be crucial in exploring and exploiting different regions of the solution space efficiently.