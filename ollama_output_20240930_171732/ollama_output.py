 # Name: Adaptive Genetic Gravitational Search with Local Random Walks
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define heuristic operators with parameters and selectors based on the provided structure
heur = [( # Search operator 1 - Genetic Mutation
    'genetic_mutation',
    {
        'scale': 1.0,
        'elite_rate': 0.1,
        'mutation_rate': 0.25,
        'distribution': 'uniform' 
    },
    'probabilistic'  # Use probabilistic selector for genetic mutations
)]



# Initialize and run the metaheuristic with the defined operators
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The metaheuristic Adaptive Genetic Gravitational Search with Local Random Walks is designed to dynamically adjust its search operators based on the complexity of the optimization problem. 
# We start with a genetic mutation operator that uses probabilistic selection, which is effective for evolving solutions in complex landscapes. 
# To enhance exploration and exploitation, we introduce gravitational search and local random walks. Gravitational search parameters are adjusted according to the dimensionality of the problem, with stronger gravity for simpler problems. 
# Local random walks adjust their probability based on problem size, favoring more exploration at smaller scales and exploitative movements as complexity increases. 
# These adaptive adjustments help in navigating the trade-off between exploration (trying new solutions) and exploitation (improving upon current best solutions), which is crucial for solving complex optimization problems efficiently.