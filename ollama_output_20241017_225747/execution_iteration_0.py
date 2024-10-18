 # Name: GravitationalSearchMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1 - Gravitational Search
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'probabilistic'
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
# The GravitationalSearchMetaheuristic employs a gravitational search algorithm where particles are attracted to the global best solution with a force proportional to their mass (defined by alpha) and distance from the center of mass. This is achieved through an iterative process where solutions evolve based on gravity-like interactions, allowing for exploration of diverse regions in the search space.
# The random_flight operator introduces a probabilistic element using Levy distribution flights that are less likely to get stuck in local minima but still maintain diversity in the population. It adjusts the scale and distribution type parameters to influence the randomness in flight paths according to user-defined strategies, enhancing exploration while allowing for some exploitation capabilities.
# Both operators are configured with probabilistic selectors to decide their application based on probability thresholds during iterations, which aids in balancing exploration and exploitation, typical challenges in metaheuristic optimization. This approach is suitable for continuous function optimizations where both global search and local refinement are needed, as demonstrated by the Rastrigin function used here.