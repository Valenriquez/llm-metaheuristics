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
# The algorithm is named Gravitational Search Optimization (GSO), which is a metaheuristic inspired by the principles of gravitational force and mass interaction. 
# We start with a population of solutions, each representing a "mass" influenced by gravity and random forces. The 'gravitational_search' operator simulates the gravitational force between masses, pulling them towards more promising regions. 
# The 'random_flight' operator introduces a probabilistic element, where solutions can move randomly scaled according to a Levy distribution or uniformly, depending on the beta parameter. This helps in escaping local minima and exploring new areas of the search space.
# Both operators are used with the selector set to 'probabilistic', meaning that their actions are governed by probabilities, enhancing exploration while still benefiting from the attractive properties of gravity when solutions are promising.