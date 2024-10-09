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
# The chosen metaheuristic is Gravitational Search Optimization (GSO), which simulates the gravitational force among particles to search for optimal solutions. 
# The first operator, 'gravitational_search', utilizes gravity as a parameter with a value of 1.0 and alpha set to 0.02. This operator operates on all elements in the population using the selector 'all'.
# The second operator is 'random_flight' which introduces random movements by scaling its scale to 1.0, with distribution set to 'levy', a type of probability function that can explore various regions of the search space effectively. Beta is set to 1.5 for this operation, adding a probabilistic element through the selector 'probabilistic'.
# These operators are selected because they balance exploration and exploitation by using gravitational forces for convergence acceleration and random flights for wide area search. The parameters are chosen based on typical values suggested in literature for GSO and PSO (Particle Swarm Optimization) related algorithms.