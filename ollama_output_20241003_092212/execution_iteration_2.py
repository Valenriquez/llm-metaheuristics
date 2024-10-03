 # Name: GravitationalSearchOptimizer
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
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
# The Gravitational Search Optimizer is a metaheuristic algorithm inspired by the principles of gravity and mass interactions in physics. 
# In this implementation, we use the gravitational search operator with predefined parameters for gravity (1.0) and alpha (0.02). 
# The selector 'all' means that this operator will be applied to all individuals in the population during each iteration. 
# This approach is used to explore the solution space by simulating the effect of gravitational forces on masses, encouraging convergence towards better solutions.