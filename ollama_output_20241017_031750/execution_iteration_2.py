 # Name: GravitationalSearchMetaheuristic
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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with two main operators: 
# Gravitational Search and Random Flight. The Gravitational Search operator uses a gravity parameter of 1.0 and an alpha value of 0.02, 
# while the Random Flight operator scales by 1.0 and uses a levy distribution with a beta factor of 1.5. 
# Both operators are configured to work with a probabilistic selector for optimal exploration in the search space. 
# The Gravitational Search operates on all possible solutions within the population, affecting them based on their gravitational forces influenced by the function's landscape. 
# This setup aims to balance between exploitation and exploration, leveraging both deterministic (Gravitational Search) and stochastic (Random Flight) methods for effective optimization.