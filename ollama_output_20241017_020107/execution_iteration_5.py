 # Name: GravitationalSearchOptimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator 1
        'gravitational_search',
        { 
            'gravity': 0.5,
            'alpha': 0.01
        },
        'metropolis'
    ),
    (  
        'random_flight',
        {
            'scale': 0.8,
            'distribution': 'levy',
            'beta': 2.0
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The algorithm chosen is Gravitational Search Optimization (GSO), which mimics the gravitational interaction among masses in space to search for optimal solutions. 
# Two main operators are used: gravitational_search and random_flight, with tailored parameters specific to GSO and other metaheuristic properties.
# Gravitational Search Operator 1 uses a lower gravity value (0.5) compared to the typical 1.0, which is adjusted for more subtle exploration of the solution space. The alpha parameter is set slightly lower than the standard 0.02 to allow for a gentler optimization process.
# The selector 'metropolis' is chosen to ensure that the selection probability follows a Metropolis-like criterion, favoring better solutions while occasionally accepting worse ones for broader search.
# Random_flight operator introduces a different dynamic by scaling down its scale (0.8) and using a levy distribution for the flight, which encourages exploration in less common areas of the solution space. The beta value is increased to 2.0, making this part of the algorithm more likely to take larger steps that are not solely based on Gaussian random walk or uniform distribution.
# This combination of operators and parameters aims to leverage both intensive local search (through gravitational_search) and broad exploration (via random_flight), providing a balance between convergence speed and global search capabilities, which is typical for metaheuristic optimization algorithms.