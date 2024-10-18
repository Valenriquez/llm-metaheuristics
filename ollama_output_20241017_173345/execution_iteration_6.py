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
        'all'
    ),
    ( # Swarm Dynamic Operator
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the Gravitational Search Algorithm (GSA) with a Swarm Dynamic algorithm to leverage both approaches' strengths in exploring diverse solution spaces. 
# The GSA uses gravitational forces to simulate mass movements, allowing it to effectively search for global minima by adjusting its parameters such as gravity and alpha. 
# The Swarm Dynamic part of the metaheuristic employs a dynamic system inspired by social swarm behavior, where particles adjust their positions based on local and global information, influenced by factors like self-confidence and swarm confidence. 
# By integrating these two operators with diverse selection strategies (all in this case), we aim to balance exploration and exploitation, ensuring that the algorithm can navigate complex landscapes efficiently.