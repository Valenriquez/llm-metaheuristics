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
# The GravitationalSearchMetaheuristic is designed to solve optimization problems using a gravitational search algorithm inspired by the physical principle of gravity. 
# The algorithm includes two main operators: gravitational_search and random_flight. 
# Gravitational search involves adjusting the positions of particles in the problem space based on their mass (or fitness), which is determined by the objective function's value. This helps to explore promising areas of the solution space more effectively.
# Random flight, a probabilistic operator, introduces randomness into the search process by using distributions such as levy or uniform distribution, allowing for broader exploration and potentially avoiding local minima. 
# Both operators are configured with specific parameters (gravity in gravitational_search and scale, distribution, beta in random_flight) to fine-tune their behavior according to the problem requirements. 
# The 'all' selector is used for both operators to ensure that they operate on all candidate solutions at every iteration of the metaheuristic process. This approach combines deterministic and stochastic elements to efficiently search for optimal solutions in complex, multi-modal optimization problems like those tackled by the Rastrigin function.