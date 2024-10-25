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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements a gravitational search metaheuristic for optimizing the Rastrigin function with two dimensions. The Gravitational Search algorithm is designed to mimic the physical principles of gravity and mass interactions, where each "particle" represents a potential solution to the problem. In this case, we use the 'gravitational_search' operator with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. The selector is set to 'all', which means all particles will be influenced by gravitational forces at each iteration. This approach allows for exploration of the search space while also considering the influence of neighboring solutions, aiming to converge towards better regions in the fitness landscape.
