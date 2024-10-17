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
    'metropolis'
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
# This code implements a metaheuristic based on the Gravitational Search Algorithm (GSA). 
# The GSA is characterized by its use of gravitational forces to guide the search towards better solutions.
# In this implementation, we have two main operators: 'gravitational_search' and 'random_flight'.
# The 'gravitational_search' operator uses parameters 'gravity' (set to 1.0) and 'alpha' (set to 0.02).
# These parameters control the strength of the gravitational force and the exploration/exploitation balance, respectively.
# The selector for this operator is set to 'metropolis', which suggests that a probabilistic selection approach will be used during the search.
# The 'random_flight' operator introduces random movements in the search space with parameters 'scale' (set to 1.0), 
# 'distribution' (set to 'levy'), and 'beta' (set to 1.5). This operator is designed to enhance the exploration of the search by introducing variability.
# The selector for this operator is set to 'probabilistic', which means that moves will be selected based on a probability distribution, promoting diversity in the search process.
# Both operators are configured according to the parameters specified in the parameters_to_take.txt file and adhere to the constraints provided by the selectors.