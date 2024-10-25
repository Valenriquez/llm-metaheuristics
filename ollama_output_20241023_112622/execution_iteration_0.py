 # Name: GravitationalSearchWithLocalRandomWalk
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1 - Gravitational Search
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
    ),
    (  
    # Search operator 2 - Local Random Walk
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'gaussian'
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the gravitational search algorithm with a local random walk to explore the solution space more effectively. The Gravitational Search operator uses gravity as a parameter to influence the movement of solutions towards better regions, while the Local Random Walk operator introduces stochasticity in the search process by allowing moves based on Gaussian distribution. Both operators are used across all iterations ('all' selector) to ensure comprehensive exploration and exploitation.
