 # Name: GravitationalSearchOptimization
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1 - Gravitational Search Optimization
        'gravitational_search',
        { 
            'gravity': 0.5,
            'alpha': 0.01
        },
        'metropolis'
    ),
    ( # Search operator 2 - Random Flight
        'random_flight',
        {
            'scale': 0.7,
            'distribution': 'levy',
            'beta': 1.0
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The algorithm used here is Gravitational Search Optimization (GSO), which mimics the gravitational force between masses in a universe to search for optimal solutions. 
# We start with an initial population generated randomly, where each solution represents a mass influenced by the gravitational forces from other masses.
# In this implementation, we set the gravity constant to 0.5 and alpha to 0.01, which are typical values as per GSO literature.
# For exploration, we use both Gravitational Search and Random Flight operators. The Gravitational Search operator uses a Metropolis selector for probabilistic decision-making during search.
# The Random Flight operator is set with a levy distribution and a beta value of 1.0, which promotes more diverse random walks in the solution space.
# These settings are based on typical parameters found in GSO literature and serve to balance exploration (through random flight) and exploitation (through gravitational attraction).
