 # Name: GravitationalSearchOptimizer
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
# The Gravitational Search Optimizer is a metaheuristic inspired by the laws of gravity and physics, which aims to optimize various functions through simulated gravitational interactions between agents in a search space. In this implementation, we use two main operators: "gravitational_search" and "random_flight". 

# The gravitational_search operator uses parameters 'gravity' (set to 1.0) and 'alpha' (set to 0.02), which control the strength of the gravitational force and its effect on agent movement, respectively. This operator is configured to operate on all agents ('all') in the population for each iteration.

# The random_flight operator simulates a random flight pattern with parameters 'scale' (set to 1.0) and 'distribution' set to 'levy', which defines the type of distribution used for this flight, providing a balance between exploration and exploitation typical of many optimization algorithms. This operator is configured to operate probabilistically ('probabilistic') to allow random movements based on these settings.

# These operators are chosen because they represent fundamental components in both gravitational search mechanisms and probabilistic movement strategies, which can effectively navigate the complex landscapes of various benchmark functions during optimization.