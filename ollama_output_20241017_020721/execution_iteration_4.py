 # Name: GravitationalSearchMetaheuristic
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
    'greedy'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is named GravitationalSearchMetaheuristic as it uses the gravitational search algorithm, which includes two main operators: gravitational_search and random_flight. 
# The gravitational_search operator has parameters gravity set to 1.0 and alpha to 0.02, while the random_flight operator has scale set to 1.0 with a distribution of levy and beta as 1.5.
# Both operators use different selectors: 'greedy' for gravitational_search and 'all' for random_flight. This combination aims to explore both deterministic greedy selection and probabilistic all-inclusive exploration provided by the random flight mechanism, which is typical in many metaheuristic algorithms. The GravitationalSearchMetaheuristic leverages these parameters to balance between local exploitation and global exploration within the search space.
# The use of 'gravitational_search' operator with specific gravity and alpha values focuses on simulating gravitational forces for particles to move towards higher fitness regions, while 'random_flight' introduces a probabilistic element with levy distribution and beta parameter to control the randomness in flight movements, contributing diverse exploration strategies not solely relying on deterministic search directions.
# The combination of these operators and parameters from the parameters_to_take.txt file ensures that the metaheuristic operates within predefined constraints and aims for optimal solution discovery by adapting its behavior according to set rules and probability distributions controlled by alpha, beta, gravity, scale, etc., which are essential in balancing convergence and diversity required for effective optimization across various benchmark functions.