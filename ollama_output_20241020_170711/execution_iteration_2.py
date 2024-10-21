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
# This code implements a Gravitational Search Metaheuristic for optimizing the Rastrigin function with 2 dimensions. The gravitational search operator is used, which involves adjusting parameters to control the strength of gravity and the scaling factor. In this case, 'gravity' is set to 1.0 and 'alpha' to 0.02. The selector 'all' means that all candidate solutions are influenced by the gravitational force. This approach aims to mimic the behavior of celestial bodies under gravitational attraction, exploring the solution space for optimal values.
