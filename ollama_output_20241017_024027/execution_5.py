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
# This metaheuristic is designed to solve optimization problems using a gravitational search approach, which mimics the physical principle of gravity for solving complex systems. The first operator used is 'gravitational_search' with parameters 'gravity' set to 1.0 and 'alpha' to 0.02. This selector ensures that all possible solutions are considered in the search space, making it suitable for exploring diverse areas of the solution space.
# The second operator is 'random_flight', which uses a probabilistic approach as specified by the 'probabilistic' selector. It has parameters 'scale' set to 1.0 and 'distribution' to 'levy'. This operator introduces randomness into the search process, allowing for exploration of new regions in the solution space that might lead to better optimization results.
# These operators together aim to balance between systematic exploration (gravitational_search) and random exploration (random_flight), which is crucial for optimizing complex functions like the Rastrigin function in a multidimensional space. The combination of gravitational search with probabilistic random flight enhances the algorithm's ability to avoid local minima and converge towards better solutions overall.