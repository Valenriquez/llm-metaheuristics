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
# The GravitationalSearchOptimizer (GSO) is a metaheuristic inspired by the laws of gravity and physics, which operates on a population of solutions that interact with each other based on their gravitational forces. In this implementation, we use two main operators from the GSO:
# 1. gravitational_search: This operator simulates the effect of gravity on particles in a multidimensional space. The parameters include 'gravity' and 'alpha', where 'gravity' represents the strength of the gravitational force, and 'alpha' is a coefficient that affects the scaling of the search space. We set 'gravity' to 1.0 and 'alpha' to 0.02. The selector for this operator is set to 'all', indicating that it will be applied to all particles in the population during each iteration.
# 2. random_flight: This operator mimics a random flight of particles, which helps in exploring different regions of the search space. Parameters include 'scale' and 'distribution' ('levy' or 'uniform'), where 'scale' determines the magnitude of the movement, and 'distribution' defines the type of distribution for this movement. Here, we set 'scale' to 1.0, and 'distribution' to 'levy'. The selector is set to 'probabilistic', meaning that the application of this operator will be governed by a probabilistic criterion during each iteration.
# These operators are chosen based on their ability to balance between exploration (random_flight) and exploitation (gravitational_search), which is typical for many metaheuristics aiming to optimize complex functions like the Rastrigin function in this case.