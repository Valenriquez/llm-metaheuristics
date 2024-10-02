 # Name: Custom Genetic Algorithm with Gravitational Search and Random Flight
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'gravitational_search',
    {
        'gravity': 1.0,
        'alpha': 0.02
    },
    'metropolis'
)] + [( # Search operator 2
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and justification:
# The metaheuristic named "Custom Genetic Algorithm with Gravitational Search and Random Flight" integrates two main operators: gravitational search and random flight to explore the solution space effectively. 
# Gravitational search, inspired by natural gravitational forces, is designed to mimic the behavior of attracting masses towards each other, promoting convergence in the optimization process. The gravity parameter controls this attraction strength, while alpha influences how quickly solutions converge to local minima or maxima based on their fitness values.
# Random flight mimics a random walk with a specific probability distribution (in this case, levy distribution) that helps in escaping from local optima and exploring new areas of the solution space more efficiently. The scale factor determines the magnitude of these exploratory steps, while beta influences the directionality of the exploration.
# By combining these two operators, we aim to balance between exploitation of promising regions (through gravitational attraction) and exploration of diverse regions (via random flight), which is expected to yield better convergence properties and more robust solutions across a variety of benchmark functions. The use of both crossover-based genetic mutation ensures that the search space is systematically explored while allowing for some degree of diversity, enhancing the algorithm's ability to avoid local minima.