 # Name: GravitationalSearchOptimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
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
# The chosen metaheuristic is Gravitational Search Optimization (GSO), which mimics the gravitational force between masses to explore the search space. 
# In this implementation, two main operators are used:
# 1. Gravitational Search: This operator uses a gravity parameter to attract particles towards better solutions and an alpha parameter to control the exploration-exploitation balance. It is applied to all agents ('all').
# 2. Random Flight: This operator simulates random movements scaled by a factor, with a distribution that can be either Levy or uniform. The beta parameter controls the randomness in this movement. Applied using a probabilistic selector for better adaptability during search.
# These operators are chosen based on their ability to balance between exploration and exploitation, which is crucial for optimizing complex functions like the Sphere function used here. Each operator's parameters are selected from the provided template according to its specific behavior: gravity and alpha in GSO control the strength of attraction and diversity, while scale and distribution in random flight adjust the step size and randomness respectively. The probabilistic selector allows these operators to dynamically influence the search based on current conditions for better convergence properties.
