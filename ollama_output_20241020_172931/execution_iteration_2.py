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
# The metaheuristic chosen is Gravitational Search Metaheuristic (GSM), which emulates the gravitational force between masses to search for optimal solutions. 
# Two main operators are implemented in this GSM version, including 'gravitational_search' and 'random_flight'.
# The 'gravitational_search' operator uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, with a selector set to 'metropolis' which suggests that the search will use probabilistic selection based on the Metropolis criterion during iterations.
# The 'random_flight' operator utilizes parameters 'scale' set to 1.0, 'distribution' set to 'levy', and 'beta' set to 1.5. This selector is 'probabilistic', indicating that this search will utilize probabilistic selection methods for guiding its random flight behavior across the search space.
# These operators are applied to solve the Rastrigin function optimization problem, which has been formatted into a standard benchmark format using the provided method from the benchmark_func module. The metaheuristic is configured with 100 iterations and runs in verbose mode for detailed output of each step's progress during execution.