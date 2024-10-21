 # Name: GravitationalSearchAlgorithmMetaheuristic
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
# This code defines a metaheuristic named GravitationalSearchAlgorithmMetaheuristic using the Gravitational Search Algorithm (GSA). The algorithm is designed to mimic gravitational interactions between particles in space, adjusting their positions based on predefined parameters for gravity and alpha. 
# The first operator used is 'gravitational_search' with specific parameters: gravity set to 1.0 and alpha to 0.02. This setting aims to simulate the effect of gravity on particles within a search space, promoting convergence towards better solutions.
# The second operator is 'random_flight', which involves random movements scaled by 1.0. These movements follow a levy distribution (type of stochastic process often used in simulations) and are biased with beta set to 1.5, enhancing exploration while maintaining some element of randomness.
# Both operators use the selector setting 'probabilistic' because they incorporate probabilistic elements: GSA relies on gravity forces that naturally have a probabilistic component, and random flights can be seen as stochastic processes influenced by probabilities. This setup allows for a balance between exploitation (using gravitational search) and exploration (random flight), which is typical in many metaheuristics to avoid getting stuck in local minima.
# The combination of these operators with their respective parameters reflects an attempt to optimize the Rastrigin function, known for its complex landscape, using principles inspired by natural phenomena.