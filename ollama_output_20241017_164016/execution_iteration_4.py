 # Name: GravitationalSearchOptimization
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
# The metaheuristic chosen is Gravitational Search Optimization (GSO). This algorithm mimics the gravitational force between masses in space, where each mass represents a candidate solution to the optimization problem. In this case, we have set up two operators within GSO: gravitational_search and random_flight. 

# The 'gravitational_search' operator is configured with parameters gravity set to 1.0 and alpha set to 0.02. This operator represents the gravitational force between masses, influencing their movement towards better solutions in the search space.

# The 'random_flight' operator uses a probabilistic selector ('probabilistic') and has parameters scale set to 1.0, distribution type set to 'levy', and beta set to 1.5. This operator introduces random movements, allowing for exploration of the solution space beyond local minima, which is beneficial in complex optimization problems like the Rastrigin function.

# Both operators are configured with parameters as specified in the parameters_to_take.txt file, ensuring that only allowed operators and parameters are used. The probabilistic selector for 'random_flight' allows it to make random moves based on a probability distribution, which is crucial for escaping local minima during search.