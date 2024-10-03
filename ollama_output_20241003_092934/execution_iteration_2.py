 # Name: PSOwithCustomParameters
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
# This code defines a metaheuristic using Particle Swarm Optimization (PSO) with custom parameters for the swarm_dynamic and random_flight operators. The PSO is chosen because it is a population-based algorithm that can balance exploration and exploitation through adjusting the positions of particles in the search space based on their own experience and the experience of neighboring particles.
# The swarm_dynamic operator uses factors to adjust the movement of particles, with self_conf and swarm_conf parameters influencing how strongly each particle adjusts its velocity towards its personal best and the global best position, respectively. The version parameter is set to 'inertial' which represents the standard PSO variant where inertia weight decreases linearly over time.
# The random_flight operator introduces a stochastic element into the search by using a distribution such as levy or gaussian for scale adjustments, with beta defining the characteristics of this distribution. This operator uses probabilistic selection to decide whether to apply the flight movement based on predefined probability thresholds.
# Both operators are combined in the heuristic list where they can dynamically influence particle positions during the PSO iterations, aiming to optimize the benchmark function Sphere(2) through a balance between deterministic and stochastic search mechanisms.