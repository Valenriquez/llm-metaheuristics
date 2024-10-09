 # Name: GravitationalSearchMetaheuristic
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
# The chosen metaheuristic is inspired by the Gravitational Search algorithm, which simulates the gravitational force between objects in space to search for optimal solutions. 
# The first operator used is 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02. This operator utilizes a deterministic approach ('all') where all particles participate actively in the search process.
# The second operator is 'random_flight' which incorporates random elements into the search by setting scale to 1.0, distribution to 'levy', and beta to 1.5. This probabilistic selection ensures that some solutions are chosen randomly based on predefined probabilities ('probabilistic').
# These operators together aim to balance exploration (gravitational force) and exploitation (random flight) in the search space for finding optimal solutions to the Rastrigin function, which is a standard benchmark problem used in optimization.