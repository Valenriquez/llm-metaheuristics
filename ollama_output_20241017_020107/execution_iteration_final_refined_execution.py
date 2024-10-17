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
    'greedy'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The GravitationalSearchMetaheuristic is designed to solve optimization problems using a gravitational search approach. 
# It begins with defining the Rastrigin function as the benchmark problem, which has two dimensions (n=2).
# Heuristically, we use one gravitational_search operator with parameters gravity set to 1.0 and alpha to 0.02.
# Additionally, a random_flight operator is introduced with scale set to 1.0, distribution as levy, and beta at 1.5.
# Both operators are configured with probabilistic selector to enhance the exploration in the solution space using probability-based mechanisms.
# The gravitational search mimics the physical law of gravitation to navigate through the search space, while random flight allows for a more explorative approach based on specified distributions.
# This combination is expected to balance between exploitation and exploration effectively, which should lead to better convergence characteristics in the optimization process.