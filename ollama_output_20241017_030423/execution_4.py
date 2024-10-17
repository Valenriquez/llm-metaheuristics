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
    'greedy'
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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm for optimization. The problem is set to Rastrigin function with dimension 2, which involves finding optimal solutions in a multi-dimensional space characterized by several local minima.
# Two main operators are employed: gravitational_search and random_flight. The gravitational_search utilizes parameters gravity (set to 1.0) and alpha (set to 0.02), while the random_flight has scale set to 1.0, distribution as levy, and beta at 1.5. These operators use selectors 'greedy' and 'probabilistic', respectively.
# The gravitational search algorithm simulates the gravitational force between masses in a multidimensional space, allowing for exploration of potential solutions. The random flight operator introduces randomness into the search process to avoid local minima by exploring new areas of the solution space. Both operators are chosen based on their ability to navigate complex landscapes and find optimal or near-optimal solutions depending on the problem's characteristics.
# This setup ensures a balanced approach between exploration (random_flight) and exploitation (gravitational_search), which is crucial for optimizing functions with multiple local minima, as seen in the Rastrigin function used here. The use of both greedy and probabilistic selectors reflects an attempt to balance convergence speed with diversity in search, though further tuning might be necessary depending on specific problem properties.