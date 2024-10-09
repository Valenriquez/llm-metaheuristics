 # Name: GravitationalSearchAlgorithm
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
# The GravitationalSearchAlgorithm (GSA) is a metaheuristic inspired by the principles of gravitational force and mass interaction. In this implementation, we use two main operators: 
# 1. Gravitational Search (gravitational_search): This operator uses parameters 'gravity' (set to 1.0) and 'alpha' (set to 0.02). The selector is set to 'metropolis', which suggests that the selection of solutions for gravitational interactions follows a probabilistic or Metropolis-like approach, allowing for exploration based on the current state and potential improvements.
# 2. Random Flight (random_flight): This operator simulates random movements influenced by parameters 'scale' (set to 1.0), 'distribution' (set to 'levy'), and 'beta' (set to 1.5). The selector is set to 'probabilistic', indicating that these random flights are probabilistic in nature, promoting diversity in the search space while allowing for a balance between exploration and exploitation.
# Both operators are designed to navigate the optimization landscape, combining gravitational forces and random perturbations to iteratively improve solution quality.