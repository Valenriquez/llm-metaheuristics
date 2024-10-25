# Name: GravitationalSearchOptimization
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
# The metaheuristic chosen is Gravitational Search Optimization (GSO), which simulates the gravitational force between masses to explore the solution space. 
# Two main operators are used in this approach: Gravitational Search and Random Flight. 
# Gravitational search applies a gravitational force based on the objective function values, promoting better solutions to move towards lower values while allowing random movement within the constraints defined by alpha and gravity parameters.
# Random flight simulates a probabilistic exploration of the solution space using a scaling factor and a distribution type (levy in this case). The beta parameter influences how much the search favors exploitation or exploration based on the problem's characteristics, making it suitable for stochastic optimization problems like Rastrigin function which is being used here.
# Both operators are selected to run over all possible solutions ('all') and 'probabilistic' selector ensures that the random flight operator operates in a probabilistic manner, allowing some solutions to be accepted based on their probability of being better than current solution without always choosing the best one. This balance between exploration and exploitation is crucial for global optimization problems like Rastrigin function which has multiple local minima.
