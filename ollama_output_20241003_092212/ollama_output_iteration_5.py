 # Name: PSOwithCustomParameters
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1 - PSO with custom parameters
        'particle_swarm',
        { 
            'inertia': 0.7,
            'cognitive': 1.5,
            'social': 1.5
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The PSO (Particle Swarm Optimization) algorithm is implemented with custom parameters to balance between exploration and exploitation. 
# Inertia weight is set to 0.7, which controls the influence of the previous velocity on the current one. 
# Cognitive parameter is set to 1.5, which represents the cognitive component influencing the particle's movement based on its own experience. 
# Social parameter is also set to 1.5, which represents the social component affecting the particle's movement due to the swarm's experience. 
# These settings aim to enhance the exploration and exploitation capabilities of the PSO algorithm by adjusting the influence of personal best and global best positions on the particles' movements.