 # Name: Adaptive Gravitational Search Metaheuristic (AGSM)
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.get_problem('sphere')  # Example function, change to your target problem
prob = fun.get_formatted_problem()

heur = [(
    'gravitational_search',
    {
        'gravity': [1.0],
        'alpha': [0.02]
    },
    'all'
), (
    'genetic_mutation',
    {
        'scale': [1.0],
        'elite_rate': [0.1],
        'mutation_rate': [0.25],
        'distribution': ['gaussian']
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The Adaptive Gravitational Search Metaheuristic (AGSM) is designed to balance the exploration and exploitation capabilities of gravitational search with the adaptive mutation mechanism typical of genetic algorithms. 
# The AGSM employs a gravitational search operator to explore the solution space, adjusting its strength according to predefined parameters. 
# Simultaneously, it utilizes genetic mutations that follow a Gaussian distribution to introduce diversity in the population, enhancing convergence towards optimal solutions without prematurely converging to local minima.
# This combination allows the metaheuristic to efficiently navigate complex landscapes and adapt to changing conditions during optimization.