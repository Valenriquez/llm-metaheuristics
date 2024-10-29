 # Name: GravitationalSearchWithMetropolisSelector
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
    'metropolis'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses a gravitational search algorithm with the metropolis selector for optimization. The gravitational search operator is configured with gravity set to 1.0 and alpha to 0.02. The random flight operator also utilizes a levy distribution, scaling factor of 1.0, and beta value of 1.5. Both operators use the metropolis selector which is suitable for probabilistic exploration in optimization problems. This setup allows for both global search capabilities of gravitational search and local improvement through the metropolis selection method, potentially leading to better convergence properties and solution quality.
# #