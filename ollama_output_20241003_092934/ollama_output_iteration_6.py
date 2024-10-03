 # Name: PSO_Optimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'swarm_dynamic',
    { 
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'uniform'
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
# The PSO_Optimization metaheuristic is designed to solve optimization problems using a particle swarm dynamics along with random flights, which are both key components of the algorithm. 
# The swarm_dynamic operator is configured with specific parameters including factor, self_conf, swarm_conf, version, and distribution to explore the solution space effectively. The factor determines how much influence each particle has on its neighborhood, while self_conf and swarm_conf control the personal and social aspects of the particles' movement respectively.
# Inertial version is chosen for the swarm dynamics which helps in balancing between exploration and exploitation by adjusting the velocity based on inertia weight. 
# The distribution parameter is set to uniform, allowing for a balanced mix of random movements within the search space.
# Additionally, the random_flight operator with parameters scale, beta, and distribution (set to levy) is included in the heuristic list. This helps in introducing stochastic elements into the algorithm by enabling particles to take random flights across the search space, aiding in diversification and exploration beyond local minima.
# The selector 'probabilistic' ensures that these operators are used with a probability-based approach, enhancing the metaheuristic's ability to adapt its strategies dynamically during optimization. This combination of dynamic swarm behavior and stochastic flight patterns should lead to efficient exploration and exploitation across various problem landscapes.