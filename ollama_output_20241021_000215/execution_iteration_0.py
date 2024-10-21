 # Name: GravitationalSearchAlgorithm
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
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
# The chosen metaheuristic is Gravitational Search Algorithm (GSA), inspired by the gravitational force in physics. 
# In this implementation, we use two operators: 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02, 
# which influences how search space is explored, and 'random_flight' with scale set to 1.0 and distribution as levy, 
# along with beta of 1.5. This combination allows for both exploration (gravitational_search) and exploitation (random_flight), 
# using a probabilistic selector to ensure diversity in the search process. The Gravitational Search Algorithm is designed for continuous optimization problems like Rastrigin function, where it aims to find the global minimum by simulating gravitational forces among mass-like particles. The use of both operators with specific parameters enhances exploration and exploitation, which are crucial for solving complex optimization tasks.
