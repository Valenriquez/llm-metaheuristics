 # Name: Adaptive Local Random Walk with Metropolis Selector
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
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (  
        'random_sample',
        {
            'selector': 'probabilistic'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic adapts the local random walk operator with a specific focus on using the Metropolis selector for exploration. This approach is motivated by the need to balance between exploitation (local search) and exploration (global search). The local random walk, configured with a probability of 0.75 and uniform distribution, allows for more focused moves near the current solution while still allowing occasional jumps across the search space. The Metropolis selector is selected as it aligns well with probabilistic sampling methods that can guide the search towards better solutions based on acceptance criteria. This combination helps in efficiently navigating the optimization landscape by ensuring both thorough exploration and targeted exploitation without getting stuck in local minima.