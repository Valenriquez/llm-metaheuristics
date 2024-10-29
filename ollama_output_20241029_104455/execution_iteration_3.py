 # Name: GravitationalSearchAndRandomFlightMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    ),
    ( # Random Flight Operator
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
# This metaheuristic combines the Gravitational Search (with gravity and alpha parameters) 
# with Random Flight (with scale, distribution of levy and beta). The Gravitational Search will be applied to all candidates, 
# while the Random Flight will use a probabilistic selector. These operators are chosen based on their ability to explore and exploit solution spaces in complex optimization problems like Rastrigin function for dimensions 2.
# Gravitational search mimics the gravitational force between masses, promoting convergence towards better solutions, 
# while random flight introduces diversity by exploring new areas of the solution space using a levy distribution or uniform noise, depending on beta setting. This hybrid approach aims to balance exploration and exploitation effectively.