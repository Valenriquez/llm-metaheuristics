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
# This code implements a gravitational search metaheuristic with the metropolis selector. 
# The gravitational_search operator is configured with gravity set to 1.0 and alpha to 0.02. 
# The selector used is 'metropolis', which suggests that the algorithm will use a probabilistic approach to accept or reject solutions based on their fitness values, potentially leading to more exploration in less promising areas.
# The Rastrigin function is chosen as it is commonly used for testing optimization algorithms due to its many local minima and complex landscape. 
# With 100 iterations, the algorithm explores the search space with a focus on finding the global minimum of the Rastrigin function.