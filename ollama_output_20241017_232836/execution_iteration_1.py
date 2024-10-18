 # Name: GravitationalSearchAlgorithmMetaheuristic
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
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is named GravitationalSearchAlgorithmMetaheuristic, inspired by the gravitational search algorithm. 
# The first operator used is 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02. 
# It uses the selector 'all', meaning it will apply this operator in every iteration of the metaheuristic process.
# The second operator is 'random_flight' which has a scale of 1.0, distribution set as 'levy', and beta at 1.5. 
# This operator utilizes the selector 'metropolis', indicating its probabilistic nature where it selects moves based on a probability threshold.
# Both operators are chosen to explore different exploration-exploitation strategies, with gravitational search focusing more on global optimization and random flight incorporating a stochastic element for diversity in the search space.