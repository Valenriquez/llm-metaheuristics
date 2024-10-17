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
        'all'
    ),
    (  
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Optimization (GSO) is a metaheuristic inspired by the principles of gravitational force in physics. 
# In this implementation, we use 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02. 
# This operator will be applied to all candidates during each iteration. Additionally, we include a 'random_flight' operator 
# with scale set to 1.0, distribution as 'levy', and beta of 1.5. The selector for this operator is set to 'probabilistic', which 
# means that the probability-based movement will be applied during each iteration according to the given parameters. This combination 
# leverages both gravitational attraction towards better solutions and random exploration, enhancing the search capabilities of the algorithm.
