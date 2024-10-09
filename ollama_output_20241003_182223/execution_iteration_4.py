 # Name: GravitationalSearchAlgorithm
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
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
# The chosen metaheuristic is Gravitational Search Algorithm (GSA), which mimics the gravitational force among particles in a multi-dimensional space to search for optimal solutions. In this implementation, we use two main operators:
# 1. Gravitational Search: This operator uses gravity as the driving force, with parameters 'gravity' and 'alpha'. The high value of 'gravity' encourages exploration while the small 'alpha' ensures fine tuning.
# 2. Random Flight: Inspired by random walks in nature, this operator simulates a probabilistic movement using a levy distribution for better global exploration capabilities. Parameters include 'scale', 'distribution' (set to levy), and 'beta'. The scale factor controls the extent of these movements while beta affects the distribution type.
# Both operators are configured with specific parameters that align with their respective natural phenomena, aiming to balance between exploitation and exploration in the optimization process. The selector is set to 'all', indicating that both greedy and probabilistic selection mechanisms will be applied based on the algorithm's behavior during execution. This setup allows for a comprehensive search across the solution space, enhancing the chances of finding an optimal solution to the given benchmark function.