 # Name: MyHeuristic
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
# The algorithm used here is a hybrid metaheuristic inspired by gravitational search and random flight mechanisms, typical of nature-inspired optimization algorithms. 
# Gravitational Search (GS) emulates the gravitational force among masses to explore the solution space, adjusting its parameters for gravity and alpha which controls the exploration vs exploitation balance. 
# Random Flight (RF), on the other hand, simulates a random walk with scaling factor that defines the step size and distribution type determines how this step is made, here set as levy distribution providing long-tail steps to escape local minima.
# The selectors used are 'all' for both operators, indicating they should be applied in every iteration of the algorithm. This choice allows for a comprehensive exploration of the search space by balancing between exploitation of potentially good solutions and exploration of new areas.
# These parameters were chosen based on typical values found in literature for gravitational and random flight inspired algorithms, aiming to achieve a balance between convergence speed and ability to avoid local minima.