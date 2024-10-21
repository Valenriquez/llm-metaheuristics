 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named GravitationalSearchMetaheuristic as it integrates the gravitational search algorithm with a focus on optimizing functions using parameters that mimic gravitational forces and interactions among particles or solutions in a multi-dimensional space. 
# The first operator used is 'gravitational_search' which simulates the behavior of celestial bodies under gravitational attraction, adjusting positions based on gravity and alpha parameters to explore and converge towards better solution regions. 
# The second operator, 'random_flight', introduces random perturbations in a controlled manner using scale, distribution (levy), and beta parameters. This helps in escaping local minima and exploring the search space more thoroughly, which is crucial for metaheuristics aiming to avoid getting stuck in poor-quality solutions. 
# The selector 'all' ensures that this probabilistic approach is applied uniformly across all iterations, while 'probabilistic' allows it to choose between exploration and exploitation based on predefined probabilities influenced by these parameters. This combination aims to balance the trade-off between convergence speed and global search capabilities inherent in many metaheuristic algorithms.