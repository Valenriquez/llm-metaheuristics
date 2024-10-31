 # Name: GravitationalSearchAlgorithmWithMetropolisSelector
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
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a metaheuristic using the Gravitational Search Algorithm (GSA) with a Metropolis selector for exploration, combined with Random Flight operator for diversification. GSA is designed to mimic gravitational forces between masses in space, adjusting positions based on gravity and alpha values to explore the solution space. The Metropolis selector allows for probabilistic acceptance of new solutions, which helps balance exploration and exploitation. The Random Flight operator introduces random jumps scaled by beta, using a levy distribution to enhance global search capabilities. This combination aims to efficiently navigate the Rastrigin function's landscape by adapting to its characteristics through these operators.
