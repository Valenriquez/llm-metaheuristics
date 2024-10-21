 # Name: GravitationalSearchOptimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    (  
        'random_flight',
        {
            'scale': 1.0,
            'distribution': "levy" or "uniform" or "gaussian",
            'beta': 1.5
        },
        'greedy' or 'all' or 'metropolis' or 'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is Gravitational Search Optimization (GSO). 
# GSO emulates the gravitational force between masses in a multidimensional space, where each mass represents a potential solution to the optimization problem.
# Heuristics used are 'gravitational_search' and 'random_flight'.
# 'gravitational_search' uses parameters: gravity = 1.0 and alpha = 0.02. This operator applies gravitational force based on the masses, promoting convergence towards better solutions.
# 'random_flight' uses parameters: scale = 1.0, distribution can be either levy, uniform, or gaussian, and beta = 1.5. This operator introduces random exploration in the search space to avoid local minima.
# Selectors used are of type 'greedy', 'all', 'metropolis', or 'probabilistic' depending on the desired behavior for exploring the solution space.
# The benchmark function used is Rastrigin, which has a dimension of 2. This function is suitable as it has multiple local minima and is multimodal, challenging the optimization algorithms to find the global minimum efficiently.