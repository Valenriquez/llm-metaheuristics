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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search and random flight operators. The gravitational search operator is configured with gravity set to 1.0 and alpha to 0.02, while the random flight operator has scale set to 1.0, distribution as levy, and beta of 1.5. Both operators use 'probabilistic' selection.
# The Rastrigin function (benchmark_func.Rastrigin(2)) is chosen as the optimization problem with a dimension of 2. A Metaheuristic object is instantiated with the defined parameters for both search operators and set to run for 100 iterations, with verbose output enabled. Finally, the best solution found is printed.
# The gravitational search operator is based on the principles of gravity simulation in physics, where 'gravity' represents the strength of attraction between particles, influencing their movement towards higher fitness regions. The random flight operator introduces a probabilistic element by considering a distribution (levy, uniform, or gaussian) and scale factor for exploration beyond local minima.
# This combination aims to leverage both deterministic gravitational forces and stochastic random flights to efficiently explore the search space of the Rastrigin function, potentially avoiding premature convergence and improving global optimization capabilities.