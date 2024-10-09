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
# The code defines a metaheuristic named GravitationalSearchMetaheuristic, which uses the gravitational search algorithm. 
# It initializes a problem using Rastrigin function with dimension 2. 
# A single heuristic is defined, where 'gravitational_search' is selected as the operator, 
# with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. The selector used is 'all', 
# indicating that this operator should be applied in each iteration of the metaheuristic.
# The algorithm runs for 100 iterations, and results are printed after optimization with best found solution.