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
# The GravitationalSearchAndRandomFlightMetaheuristic combines two metaheuristic operators, 
# Gravitational Search and Random Flight, to explore the solution space. 
# Gravitational search is designed to mimic the gravitational force between masses, 
# with a gravity parameter controlling the strength of this force, and alpha determining the rate of convergence. 
# The Random Flight operator uses a random walk based on a specified distribution (levy in this case), scaled by a factor, 
# which helps in escaping local minima. This combination aims to balance exploration and exploitation effectively.
# The selector 'probabilistic' is chosen for the Random Flight operator as it operates probabilistically, 
# allowing particles or solutions to move randomly based on predefined probabilities.