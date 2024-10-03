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
        'selector': 'greedy' or 'all' or 'metropolis' or 'probabilistic'
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed as a Random Search algorithm tailored for ill-structured global optimization problems where the objective function may exhibit nonconvexity, nondifferentiability, and discontinuity. The search operator employs either Gaussian or Levy distribution to explore the solution space, ensuring diversity in exploration. The selector determines how the population evolves, using 'greedy' selection which keeps the best individuals, 'all' which includes all candidates, 'metropolis' for probabilistic acceptance based on energy levels, or 'probabilistic' which follows a stochastic process. This approach allows for effective navigation through complex landscapes without being constrained by specific mathematical properties of the problem functions.
# The implementation uses random search methods with diverse exploration strategies to handle problems that do not conform to conventional optimization paradigms, ensuring flexibility and adaptability in tackling various ill-structured global optimization challenges.