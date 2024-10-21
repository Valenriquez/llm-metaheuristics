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
    'metropolis'
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
# The algorithm is named Gravitational Search Optimization (GSO), which is a metaheuristic inspired by the laws of gravity and physical interactions. 
# In this implementation, we have two main search operators: 'gravitational_search' and 'random_flight'. 
# 'Gravitational search' uses parameters 'gravity' and 'alpha' to simulate gravitational forces between particles in the search space. 
# This operator is combined with a 'metropolis' selector for probabilistic selection of solutions based on their fitness values.
# The second operator, 'random_flight', involves setting scale to 1.0 and distribution to 'levy'. 
# This represents random flights in the search space, where 'levy' defines the flight pattern influenced by Levy distributions. 
# It is used with a 'probabilistic' selector for exploration based on probability of selection.
# These operators are designed to balance between global and local explorations, which should help in finding better solutions for the Rastrigin function optimization problem.