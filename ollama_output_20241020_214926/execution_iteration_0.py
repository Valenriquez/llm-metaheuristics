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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm, which is characterized by its gravity parameter influencing the movement of particles in the solution space. The first operator is set to be applied to all elements with specific parameters for gravity and alpha. The second operator uses random flight behavior with specified scale and distribution type (levy) along with beta value affecting the probabilistic selection. Both operators are integrated into a metaheuristic framework, which runs for 100 iterations, aiming to find the optimal solution for the Rastrigin function in two dimensions.
# The gravitational search algorithm is inspired by the concept of gravity and mass interaction influencing particle movement in optimization problems. By setting 'gravity' to 1.0 and 'alpha' to 0.02, it encourages a balance between exploration (low alpha) and exploitation (high alpha), which could be suitable for complex functions like Rastrigin's with multiple peaks and valleys.
# The random flight operator introduces stochastic elements by using a levy distribution, known for its long tails that can effectively explore remote areas of the search space. This is beneficial when dealing with multimodal functions where solutions might be scattered across various regions.
# The probabilistic selector 'probabilistic' ensures that some decisions are made based on probability rather than strict rules, which could help in avoiding local minima by occasionally taking random leaps in the solution space.
