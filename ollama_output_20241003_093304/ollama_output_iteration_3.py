 # Name: GravitationalSearchAlgorithmMetaheuristic
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
# The Gravitational Search Algorithm (GSA) is inspired by the gravitational force in physics, where objects are attracted to more massive ones. In this metaheuristic, we use two main operators: 'gravitational_search' and 'random_flight'. 

# The 'gravitational_search' operator uses parameters 'gravity' and 'alpha', which represent the strength of the gravitational force and a scaling factor, respectively. These are set to 1.0 and 0.02 based on the provided parameters in parameters_to_take.txt. The selector is set to 'metropolis', as it fits well with probabilistic search strategies.

# The 'random_flight' operator simulates random movements of particles, influenced by a scale factor and distribution type ('levy'). Parameters include 'scale', 'distribution', and 'beta'. 'Scale' is set to 1.0 for uniform scaling, and 'distribution' is 'levy' which supports non-uniform random walks characteristic of complex terrain search. The selector here is 'probabilistic' as it aligns with the stochastic nature of this operator.

# Both operators are carefully chosen from parameters_to_take.txt to ensure they match the probabilistic and gravitational physics principles, respectively. This setup aims to balance exploration and exploitation in the optimization process, suitable for continuous function optimizations like those evaluated by benchmark functions such as Sphere.