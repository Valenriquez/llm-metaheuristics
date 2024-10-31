 # Name: GravitationalSearchMetaheuristic
# Code:
import sys
from pathlib import Path

# Move up one level to the llm-metaheuristics directory
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the Gravitational Search algorithm with specified parameters for search operators. The Rastrigin function is chosen as the benchmark problem, which is formatted appropriately for the metaheuristic framework. Two main search operators are included in this implementation: gravitational_search and random_flight.
# The gravitational_search operator uses gravity (set to 1.0) and alpha (set to 0.02) as its parameters, while the selector type is set to 'all'. This means that all possible selections from the population will undergo a gravitational pull during each iteration.
# The random_flight operator utilizes scale (also set to 1.0), distribution ('levy'), and beta (set to 1.5). It operates with a probabilistic selection, which is indicated by 'probabilistic' in the selector field. This helps in exploring diverse areas of the search space based on the defined distribution.
# Both operators are chosen for their ability to balance exploration and exploitation, which should be beneficial for optimizing complex multimodal functions like the Rastrigin function. The Gravitational Search algorithm is known for its effectiveness in solving optimization problems with multiple local minima, making it suitable for this task.