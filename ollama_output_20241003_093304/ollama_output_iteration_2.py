 # Name: PSO_Metaheuristic
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
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'gaussian'
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The PSO_Metaheuristic is designed to optimize the Sphere function using a combination of swarm dynamics and local random walks. 
# Swarm dynamics are used with parameters factor set to 0.7, self_conf to 2.54, and swarm_conf to 2.56, version set to inertial, and distribution set to uniform.
# Local random walks use a probability of 0.75 and scale set to 1.0 with gaussian distribution for exploration. The selector 'probabilistic' allows these operators to be applied probabilistically during the optimization process, promoting diverse exploration without getting stuck in local minima. This combination aims to balance between global exploration and local exploitation, enhancing the overall performance of the metaheuristic.