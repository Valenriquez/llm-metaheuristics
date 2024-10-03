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
    'random_search',
    {
        'distribution': 'gaussian' or 'levy',
    },
    'all'
    ),
    ( # Search operator 2
    'random_search',
    {
        'distribution': 'gaussian' or 'levy',
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed as a Random Search algorithm tailored for ill-structured global optimization problems, where the objective function may be nonconvex, nondifferentiable, and possibly discontinuous over a continuous, discrete, or mixed continuous-discrete domain. 
# Two random search operators are implemented with different distributions (gaussian and levy) to explore the solution space effectively. The 'all' selector is used to apply these operators across all possible solutions in each iteration. This approach ensures comprehensive exploration of the problem space while adapting the search strategy based on the specific characteristics of the distribution for better convergence towards an optimal solution.