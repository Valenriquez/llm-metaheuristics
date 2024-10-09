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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm, which is inspired by the physics of celestial bodies influenced by gravity. The first operator used in this case is the gravitational search with parameters for gravity set to 1.0 and alpha to 0.02. The selector for this operator is set to 'metropolis', which suggests a probabilistic approach where decisions are made based on probabilities, potentially guiding the search towards better solutions more often than not but allowing occasional random jumps that might help escape local minima.
# The second operator used in this code snippet is a random flight algorithm with parameters including scale set to 1.0 and distribution type 'levy'. This operator uses a probabilistic selector named 'probabilistic', which means it will follow a stochastic approach, choosing moves based on their probability of leading to better solutions according to some probabilistic model or rule.
# These two operators together provide a combination of directed search (gravitational search) and exploration (random flight), which should help in efficiently exploring the solution space for optimization problems like Rastrigin function minimization by leveraging both certainty-based and probabilistic decision-making mechanisms typical of metaheuristic approaches.