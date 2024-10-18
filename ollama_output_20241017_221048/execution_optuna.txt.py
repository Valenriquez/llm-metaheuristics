 # Name: Custom Genetic Algorithm with Rastrigin Function
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ('genetic_crossover', { 
        'pairing': 'tournament_2_100',  
        'crossover': 'uniform', 
        'mutation': 'gaussian',  # Added genetic mutation
        'mutation_rate': trial.suggest_float('mutation_rate', 0.01, 0.1),  # Suggested parameter from optuna
        'mating_pool_factor': trial.suggest_float('mating_pool_factor', 0.1, 0.9)  
    }, 'all'),
    
    ('swarm_dynamic', {
        'factor': trial.suggest_float('factor', 0.4, 0.9),
        'self_conf': trial.suggest_float('self_conf', 1.5, 3.0),
        'swarm_conf': trial.suggest_float('swarm_conf', 1.5, 3.0),
        'version': 'inertial', 
        'distribution': 'uniform' 
    }, 'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a custom genetic algorithm with Rastrigin function optimization. 
# Two main operators are used: genetic_crossover for crossover operations and swarm_dynamic for dynamic swarming behavior.
# Genetic crossover includes an additional 'mutation' parameter set to 'gaussian', which introduces random changes in the population based on Gaussian distribution.
# The mutation rate is suggested using optuna, allowing for variation within a specified range (0.01 to 0.1). This helps in exploring different levels of mutation that might lead to better fitness values.
# Swarm_dynamic includes several parameters such as 'factor', 'self_conf', and 'swarm_conf' which are suggested using optuna, enabling the algorithm to adapt its behavior dynamically based on these configurations.
# The genetic crossover operation is paired with a genetic mutation to ensure comprehensive exploration of the solution space during optimization.