 # Name: GravitationalSearchAlgorithm
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
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    (  
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy' or 'uniform' or 'gaussian',
            'beta': 1.5
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 3
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform' or 'gaussian' or 'levy'
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 4
        'random_sample',
        {},
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 5
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    ( # Search operator 6
        'swarm_dynamic',
        {
            'factor': 0.7 or 1.0,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial' or 'constriction',
            'distribution': 'uniform' or 'gaussian' or 'levy'
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic optimization using the Gravitational Search Algorithm (GSA). 
# The algorithm consists of six search operators, each with specific parameters as defined in the parameters_to_take.txt file. 
# The GSA is designed to explore the problem space by simulating the gravitational force between masses, where each operator represents a different method for exploring this space.
# The Gravitational Search Algorithm (GSA) is an optimization algorithm inspired by the principles of physics, specifically the law of gravity and mass interactions. In this implementation, six distinct search operators are used: 
#   - Gravitational Search (gravitational_search): Simulates the gravitational force between masses to explore the solution space.
#   - Random Flight (random_flight): Emulates random movements in a scaled manner, with options for different distributions.
#   - Local Random Walk (local_random_walk): Conducts local exploration based on a probability and scale parameter, allowing for uniform or Gaussian distribution.
#   - Random Sample (random_sample): Selects solutions randomly from the population.
#   - Spiral Dynamic Search (spiral_dynamic): Moves towards areas of higher density using a spiral pattern with specified radius and angle parameters.
#   - Swarm Dynamics (swarm_dynamic): Explores the solution space like a swarm, adjusting its movement based on self-confidence and swarm confidence factors, with options for different distributions.
# The selectors can be 'greedy', 'all', 'metropolis', or 'probabilistic' to determine how these operators are applied during each iteration of the algorithm. This setup allows for a comprehensive exploration of the solution space by combining diverse search strategies.