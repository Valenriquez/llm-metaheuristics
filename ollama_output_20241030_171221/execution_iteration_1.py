 # Name: GravitationalSearchOptimizer
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
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The Gravitational Search Optimizer (GSO) is a metaheuristic inspired by the principles of gravitational force and mass interactions. In this implementation, we use the 'gravitational_search' operator with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. The selector used is 'all', which means all particles will undergo gravitational search at each iteration. This approach allows for exploration of the solution space by simulating the effect of gravity on masses, promoting convergence towards better solutions while preventing premature convergence.
