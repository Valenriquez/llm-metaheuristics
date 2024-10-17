 # Name: GravitationalSearchOptimizer
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
# The GravitationalSearchOptimizer (GSO) is a metaheuristic inspired by the physics of gravitational forces between masses. In this implementation, we use two operators: 'gravitational_search' and 'random_flight'. 
# The 'gravitational_search' operator uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. It operates on all solutions in the population ('all'). 
# The 'random_flight' operator employs a scale of 1.0, with the distribution type set to 'levy'. This is a probabilistic method as it includes a stochastic component determined by the levy distribution. The selector for this operator is set to 'probabilistic', which means that the selection process involves probabilities, potentially leading to more diverse exploration in the search space compared to deterministic methods like greedy or metropolis.
# These operators are chosen based on their capability to explore and exploit different parts of the solution space, with GSO being well-suited for continuous optimization problems such as the Rastrigin function used here. The probabilistic nature of 'random_flight' helps in avoiding local minima by allowing random jumps that might lead to better solutions.