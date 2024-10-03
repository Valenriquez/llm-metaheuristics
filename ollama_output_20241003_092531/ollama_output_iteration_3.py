 # Name: GravitationalSearchOptimizer
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
# The GravitationalSearchOptimizer is a metaheuristic inspired by the laws of gravity and motion, where particles or solutions are attracted to better regions based on their fitness values. In this case, we use gravitational search with parameters set as follows:
# 1. Gravity (gravity) is set to 1.0, which represents the strength of the gravitational force.
# 2. Alpha (alpha) is set to 0.02, affecting how quickly solutions converge towards better regions.
# Additionally, we include a random flight operator with specific parameters:
# 1. Scale (scale) is set to 1.0, determining the magnitude of the random step.
# 2. Distribution (distribution) is 'levy', which introduces randomness following a levy distribution, aiding in exploration of diverse regions.
# The selector for both operators is set to 'all' and 'probabilistic' respectively, allowing for probabilistic exploration based on fitness values while maintaining gravitational attraction towards better solutions. This combination aims to balance between exploitation and exploration effectively, leveraging the strengths of both search mechanisms.