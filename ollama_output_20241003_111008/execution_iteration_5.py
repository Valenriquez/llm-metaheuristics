 # Name: PSOGravitationalSearchHybrid
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # PSO Search Operator
        'particle_swarm',
        { 
            'population_size': 50,
            'inertia_weight': 0.7,
            'cognitive_coefficient': 1.5,
            'social_coefficient': 1.5
        },
        'all'
    ),
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 0.8,
            'alpha': 0.1
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The PSOGravitationalSearchHybrid algorithm combines the particle swarm optimization (PSO) with gravitational search to leverage both methods' strengths. 
# PSO is used for global exploration with parameters set to encourage balance between exploration and exploitation, while gravitational search introduces local search capabilities through the 'metropolis' selector which promotes convergence towards better solutions.
# Both operators are applied iteratively according to their selectors, allowing for a hybrid approach that benefits from diverse search patterns without over-reliance on either method alone.