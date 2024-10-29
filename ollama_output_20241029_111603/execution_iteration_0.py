 # Name: GravitationalSearchWithMetropolisSelector
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
        'metropolis'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The algorithm is designed to use Gravitational Search with a Metropolis selector for optimization of the Rastrigin function in 2 dimensions.
# The gravitational search operator is configured with gravity set to 1.0 and alpha to 0.02, which are parameters known for their effectiveness in exploring diverse solution spaces.
# The selector 'metropolis' is chosen because it allows for a balance between exploration and exploitation, which can lead to better convergence properties when dealing with complex objective functions like the Rastrigin function.
# This setup aims to leverage the strengths of both gravitational search and Metropolis selection in navigating the optimization landscape efficiently.