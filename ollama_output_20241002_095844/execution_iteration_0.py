 # Name: MyCustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Using greedy selector for all operators to ensure a balanced exploration and exploitation
heur = [(
    'genetic_mutation',  # Search operator 1
    {
        'scale': 0.5,
        'elite_rate': 0.2,
        'mutation_rate': 0.3,
        'distribution': 'gaussian'
    },
    'greedy'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic named MyCustomMetaheuristic is designed using a greedy selector across all operators for consistency in the search strategy. 
# The genetic_mutation operator is chosen with specific parameter settings, including scale, elite rate, mutation rate, and distribution type. 
# These parameters are selected based on typical values observed to be effective in similar optimization problems, ensuring a balance between exploration (high mutation scale) and exploitation (low mutation rate).
# The Rastrigin function is used as the benchmark problem, which has two dimensions, suitable for visualization and analysis of multi-modal functions. 
# The metaheuristic runs for 150 iterations to allow sufficient time for convergence while preventing excessive computation. This setting allows exploration of various regions in the search space before converging on an optimal solution.