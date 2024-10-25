 # Name: GravitationalSearchAlgorithm
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
# The algorithm GravitationalSearchAlgorithm is implemented using two main operators: gravitational_search and random_flight. 
# Gravitational search employs gravity as a parameter to simulate the movement of celestial bodies, while random flight uses scale and distribution to explore the solution space in a probabilistic manner.
# Both operators use 'all' as the selector, meaning they will be applied to all particles during each iteration. This approach aims to balance exploration and exploitation efficiently.
# The GravitationalSearchAlgorithm is chosen because it combines both local search capabilities (through gravitational_search) and global exploration (through random_flight), which are typical for metaheuristic algorithms. 
# The parameters gravity (0.1-1.0) and alpha (0.01-0.2) in the gravitational_search operator, as well as scale (0.1-3.0), distribution ('levy' or 'uniform'), and beta (0.5-3.0) in the random_flight operator are selected based on typical values found in literature for similar algorithms. 
# These parameters have been chosen to ensure a balance between convergence speed and accuracy, as well as to prevent premature convergence by allowing exploration of various regions of the search space.