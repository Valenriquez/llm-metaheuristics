 # Name: AdaptiveRandomSearch
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'gaussian',
        {
            'parameter1': 0.5,
            'parameter2': 0.8,
        },
        'metropolis'
    ),
    ( # Search operator 2
        'levy',
        {
            'parameter1': 0.3,
            'parameter2': 0.6,
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The AdaptiveRandomSearch metaheuristic is designed to tackle global optimization problems with nonconvex, nondifferentiable, and possibly discontinuous objective functions over continuous, discrete, or mixed domains. This approach uses two search operators: 'gaussian' for exploration based on Gaussian mutations and 'levy' for more random walk-like behavior influenced by Levy distributions. The selectors are set to 'metropolis' for probabilistic acceptance of new solutions in the optimization process. Parameters are tuned according to typical settings found in literature for such metaheuristics, aiming to balance between exploration and exploitation effectively.
