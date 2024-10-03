 # Name: RandomSearchMetaheuristic
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
        'scale': 0.5,
        'distribution': 'normal'
    },
    'greedy'
    ),
    ( # Search operator 2
    'levy',
    {
        'scale': 1.0,
        'distribution': 'levy'
    },
    'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named RandomSearchMetaheuristic, inspired by the random search method typically used for global optimization problems with nonconvex, nondifferentiable, or possibly discontinuous objective functions. Two search operators are implemented here: 'gaussian' for Gaussian mutation with a normal distribution and 'levy' for Levy flight mutation with a levy distribution. The selector is set to 'greedy' for the first operator and 'all' for the second, which means each will be applied in turn during the optimization process. This setup allows exploration of different search behaviors within the framework of random search algorithms.