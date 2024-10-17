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
# This metaheuristic is designed based on the Gravitational Search algorithm, which mimics the gravitational force between masses to search for optimal solutions. The parameters used are 'gravity' set to 1.0 and 'alpha' set to 0.02 as specified in the parameters_to_take.txt file. The selector 'all' is chosen to apply this operator across all possible selections, ensuring a comprehensive exploration of the solution space. This approach allows for both global and local search capabilities by simulating gravitational interactions among hypothetical masses.
