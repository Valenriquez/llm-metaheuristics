 # Name: PSOWithCustomOperators
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1: PSO with inertia weight and cognitive/social acceleration coefficients
        'particle_swarm',
        { 
            'inertia_weight': 0.7,
            'cognitive_acceleration': 1.5,
            'social_acceleration': 1.5
        },
        'all'
    ),
    ( # Search operator 2: Gravitational search with custom gravity and alpha parameters
        'gravitational_search',
        { 
            'gravity': 0.9,
            'alpha': 0.5
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The PSOWithCustomOperators metaheuristic uses a particle swarm optimization (PSO) with custom parameters for inertia weight and cognitive/social acceleration coefficients, along with gravitational search to explore the solution space. 
# PSO is effective in balancing exploration and exploitation by adjusting its parameters; here, an increased social_acceleration helps in better global exploration while maintaining local fine-tuning through the inertia_weight. 
# Gravitational search introduces a novel approach with custom gravity and alpha values which are tuned for probabilistic selection of candidate solutions during iterations, enhancing both convergence speed and accuracy.
# Both operators leverage different mechanisms to navigate the problem space: PSO relies on swarm intelligence for dynamic movement based on personal best and neighborhood best positions; gravitational search mimics physical laws to attract particles towards better regions in the solution landscape. 
# The combination of these two approaches aims to benefit from diverse exploration techniques, ensuring a robust optimization across various scenarios without becoming overly biased towards either local minima or random noise.