 # Name: GravitationalSearchAlgorithmMetaheuristic
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
# This code implements a Gravitational Search Algorithm (GSA) metaheuristic for optimizing the Rastrigin function with two dimensions. The GSA is characterized by its gravitational constant (gravity) and alpha parameter which influences the strength of the gravitational force. Here, both parameters are set to default values as per the reference: gravity=1.0 and alpha=0.02.
# The selector 'all' is chosen because the Gravitational Search Algorithm operates on all candidate solutions in each iteration, not just a subset or following a specific selection strategy. This approach allows for global exploration of the search space to find better solutions by mimicking gravitational interactions among masses.
