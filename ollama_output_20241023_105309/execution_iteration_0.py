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
# This code defines a metaheuristic using the Gravitational Search Algorithm (GSA) with a Metropolis selector for exploration. The GSA is designed to mimic gravitational forces between masses, where 'gravity' parameter controls the strength of the force, and 'alpha' affects the scaling factor. A random flight operator is introduced using a probabilistic selector, which explores solutions based on the specified distribution ('levy', in this case). The Gravitational Search Algorithm with Metropolis selection aims to balance exploitation and exploration by allowing moves that may lead to worse solutions with a certain probability (controlled by 'alpha'), promoting diverse search paths.
