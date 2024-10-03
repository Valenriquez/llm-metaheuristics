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
    },
    'greedy'
    ),
    ( # Search operator 2
    'levy',
    {
        'scale': 0.7,
    },
    'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic chosen is named RandomSearchMetaheuristic. It uses two search operators: gaussian and levy, both with their respective parameters scale set to 0.5 for gaussian and 0.7 for levy. These operators are applied using the selectors greedy and all respectively. This approach is justified as it leverages random search algorithms which are effective for ill-structured global optimization problems where the objective function may be nonconvex, nondifferentiable, and possibly discontinuous over a continuous, discrete, or mixed continuous-discrete domain. The use of gaussian and levy distributions allows for exploration in different ways (Gaussian explores around current solutions with some variability controlled by scale; Levy flight is more exploratory but less biased). The greedy selector focuses initial iterations on improving the best solution found so far, while all selects either always uses the new operator result or considers both results.
