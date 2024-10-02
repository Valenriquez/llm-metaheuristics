 # Name: CustomMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operators with their respective parameters and selectors
heur = [
    (
        'genetic_mutation',
        {
            'parameters': {
                'scale': 1.0,
                'elite_rate': 0.1,
                'mutation_rate': 0.25,
                'distribution': 'uniform'
            }
        },
        'probabilistic'
    ),
    (
        'greedy',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic named CustomMetaheuristic is designed to solve the Rastrigin function optimization problem with a combination of genetic mutation and greedy search operators. 
# The genetic mutation operator uses parameters for scale, elite rate, mutation rate, and distribution (set to uniform). The selector probabilistic allows this operator to be used in a probabilistic manner during the search process.
# The greedy search is included as another operator without any specific parameters, using the all selector which means it will apply the greedy strategy across the entire population. 
# This combination aims to leverage both exploration provided by genetic mutation and exploitation strategies employed by the greedy approach for efficient optimization of the Rastrigin function.