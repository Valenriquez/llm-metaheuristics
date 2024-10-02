 # Name: Hybrid Genetic Algorithm with Gravitational Search and Spiral Dynamic Optimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'genetic_mutation',
    {
        'scale': 1.0,
        'elite_rate': 0.1,
        'mutation_rate': 0.25,
        'distribution': 'uniform'
    },
    'probabilistic'
)] + [( # Search operator 2
    'gravitational_search',
    {
        'gravity': 1.0,
        'alpha': 0.02
    },
    'greedy'
)] + [( # Search operator 3
    'spiral_dynamic',
    {
        'radius': 0.9,
        'angle': 22.5,
        'sigma': 0.1
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The hybrid metaheuristic combines genetic mutation for exploring the solution space, gravitational search to mimic the effect of gravity on particles in a multi-dimensional space, and spiral dynamic optimization to introduce local perturbations around potentially optimal regions. This approach leverages diverse mechanisms to efficiently explore and exploit the problem landscape, ensuring a balance between global and local search capabilities. The probabilistic selector is used for genetic_mutation and gravitational_search to ensure that both exploration and exploitation are dynamically adjusted during runtime.