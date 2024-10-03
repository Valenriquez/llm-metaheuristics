 # Name: PSO_GravitationalSearch
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1: GravitationalSearch
        'gravitational_search',
        { 
            'gravity': 0.5, # Adjusted gravity for faster convergence
            'alpha': 0.2   # Increased alpha to encourage more exploration
        },
        'all'
    ),
    ( # Search operator 2: PSO (Particle Swarm Optimization)
        'swarm_dynamic',
        {
            'factor': 0.8, # Adjusted factor for balance between exploration and exploitation
            'self_conf': 1.5, # Reduced self-confidence for more social influence
            'swarm_conf': 1.5, # Reduced swarm-confidence for better global search
            'version': 'inertial', # Keeping the inertial version for balance
            'distribution': 'gaussian' # Gaussian distribution for velocity initialization
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# PSO_GravitationalSearch combines the strengths of Particle Swarm Optimization (PSO) and Gravitational Search (GS). 
# The GravitationalSearch operator is configured with a lower gravity for faster convergence but increased alpha to encourage exploration, enhancing global search capabilities.
# The swarm_dynamic uses a factor that balances between exploration and exploitation by reducing self-confidence and swarm-confidence. It also employs the inertial version of PSO with a Gaussian distribution for velocity initialization to ensure diversity in movement patterns.
# Both operators are set to run iteratively ('all') through the optimization process, allowing for a comprehensive search across the solution space, which is crucial for finding the optimal parameters that lead to better fitness values according to the Sphere function benchmark.