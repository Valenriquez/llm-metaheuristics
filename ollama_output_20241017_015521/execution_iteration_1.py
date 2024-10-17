 # Name: GravitationalSearchAlgorithm
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
# The algorithm is named GravitationalSearchAlgorithm as it incorporates the concept of gravitational force in its search mechanism, similar to how particles interact with each other under gravity within a system. This approach aims to simulate the natural laws of physics for solving optimization problems by mimicking the behavior of particles moving towards masses or attractors.
# The first operator used is 'gravitational_search', which takes parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. This configuration aims to simulate a system where stronger gravity promotes convergence towards better solutions, while the alpha parameter controls the scaling of this effect.
# The selector for the first operator is set to 'all', indicating that each particle in the population will use this search method during every iteration.
# The second operator is 'random_flight' with parameters 'scale' set to 1.0 and 'distribution' as 'levy'. This operator uses a probabilistic approach where particles can move in directions influenced by a levy distribution, which has heavy tails that allow for more exploration of the search space compared to uniform or gaussian distributions.
# The selector for the second operator is set to 'probabilistic', meaning this method will be applied with a probability during each iteration, encouraging both exploitation and exploration depending on the problem's requirements.
# This combination aims to balance between thorough exploration and efficient exploitation of known areas in the search space, which should lead to better convergence towards an optimal solution for the given Rastrigin function.