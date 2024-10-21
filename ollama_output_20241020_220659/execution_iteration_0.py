 # Name: GravitationalSearchAndSwarmDynamicMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (   # Swarm Dynamic Operator
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines Gravitational Search with Swarm Dynamic optimization techniques to solve the Rastrigin function problem. 
# The Gravitational Search operator uses a gravity parameter of 1.0 and an alpha value of 0.02, while the Swarm Dynamic operator employs a factor of 0.7 for self-confidence, a swarm confidence of 2.56, and uses the inertial version with a uniform distribution.
# The Gravitational Search is set to use a greedy selector to guide its search towards better solutions, while the Swarm Dynamic utilizes a probabilistic selector to balance exploration and exploitation based on the probability parameter. This combination aims to leverage both techniques' strengths in different ways to efficiently navigate the complex landscape of the Rastrigin function.