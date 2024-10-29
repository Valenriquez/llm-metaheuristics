 # Name: GravitationalSearchWithMetropolisSelector
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
        'distribution': "levy",
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
# The metaheuristic used in this code is Gravitational Search with Metropolis Selector. 
# This method incorporates both gravitational search for exploration and random flight for exploitation, 
# combined with the probabilistic selector to balance between local and global searches. 
# Gravitational search simulates the gravitational force acting on particles to find the optimal solution, 
# while random flight introduces diversity by simulating random movements influenced by specified distributions. 
# The Metropolis selector is chosen as it allows for a controlled exploration-exploitation trade-off by accepting or rejecting moves based on a probability distribution. 
# This combination helps in efficiently navigating through the search space to find better solutions, which is crucial for optimization problems like Rastrigin's function.