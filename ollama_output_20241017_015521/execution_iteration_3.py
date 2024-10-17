 # Name: GravitationalSearchAndRandomFlightMetaheuristic
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
    ( # Random Flight Operator
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
# This metaheuristic combines the Gravitational Search (gravity-driven optimization) with Random Flight (exploration of search space). The gravitational_search operator uses a gravity parameter to simulate the attraction between particles, while the random_flight operator explores the search space using different distributions. Both operators use 'all' as the selector for continuous exploration and exploitation throughout the iterations. This combination is intended to leverage both global and local search capabilities, potentially leading to better convergence properties compared to using either method alone.
# The Gravitational Search part of this metaheuristic aims to mimic the gravitational force among particles in a system, where each particle's position represents a candidate solution to the optimization problem. The 'gravity' parameter controls the strength of the attraction, while 'alpha' adjusts the effect of local search versus global search. This setup is expected to help in escaping local minima and exploring the entire search space effectively.
# On the other hand, the Random Flight operator introduces random perturbations into the particle positions based on specified distributions ('levy', 'uniform', or 'gaussian'). The scale parameter controls the amplitude of these random jumps, which can be adjusted according to the specific problem characteristics. The 'beta' parameter influences the distribution type and thus affects how widely the solutions are explored in each iteration.
# By combining these two operators with different selectors ('all' for probabilistic exploration), we aim to balance between thorough search (using gravitational attraction) and random jumps (as a means to escape local minima). This approach is expected to provide a robust method for solving complex optimization problems, where both global convergence and the avoidance of premature convergence are crucial.