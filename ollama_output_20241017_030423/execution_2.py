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
# The chosen metaheuristic is Gravitational Search Optimization (GSO), which mimics the gravitational force between masses to search for the optimal solution. The first operator used in this implementation is 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02. This operator will be applied to all agents as specified by the selector 'all'.
# The second operator is 'random_flight', which uses a random flight mechanism. It has parameters scale set to 1.0, distribution type as 'levy', and beta value of 1.5. The selector for this operator is 'probabilistic', meaning it will be used with a probabilistic approach during the search process.
# These operators are chosen based on their ability to explore diverse areas of the solution space through gravitational force-inspired mechanism in the first case, and random flight strategy in the second case, which complements each other for effective global optimization. This combination is justified by the need to leverage both deterministic and probabilistic exploration techniques within a single metaheuristic framework.