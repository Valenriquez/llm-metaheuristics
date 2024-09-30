 # Name: HybridMetaHeuristic
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
        'distribution': "gaussian"
    },
    'all'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# The metaheuristic is named HybridMetaHeuristic because it combines the genetic mutation operator with a specific distribution for mutation, which in this case is Gaussian. 
# This choice is based on the premise that Gaussian distribution can effectively explore and exploit different regions of the search space due to its properties of being centered around a mean value with decreasing variance as one moves away from the center. The use of genetic crossover and mutation together aims to balance between exploration (mutation) and exploitation (crossover), which is typical in hybrid metaheuristics.
# The specific parameters for genetic_mutation are chosen based on their potential to influence the mutation process in a way that can lead to better convergence properties, such as faster convergence or more diverse solutions around the optimum.
