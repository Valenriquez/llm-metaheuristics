 # Name: GravitationalSearchOptimization
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
# This code implements a metaheuristic named Gravitational Search Optimization (GSO). 
# The GSO is inspired by the gravitational force acting between masses in physics, where each "mass" represents a potential solution to the problem.
# In this implementation, we use two main operators: 'gravitational_search' and 'random_flight'. 
# The 'gravitational_search' operator uses parameters gravity (1.0) and alpha (0.02) to adjust the strength of the gravitational force between masses. 
# The 'random_flight' operator introduces a random flight capability with scale (1.0), distribution ('levy'), and beta (1.5). 
# Both operators use the selector 'all', which means they will be applied in each iteration to all potential solutions, allowing for exploration of the solution space.
# The benchmark function used is Rastrigin's with a dimension of 2. GSO aims to find an optimal solution by iteratively adjusting the positions of masses according to the laws of gravity and random flight dynamics.
