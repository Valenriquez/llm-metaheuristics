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
# The chosen metaheuristic is Gravitational Search Optimization (GSO). This method simulates the gravitational force among particles to search for optimal solutions. 
# In this case, we have defined two operators within GSO: gravitational_search and random_flight.
# The 'gravitational_search' operator uses parameters gravity and alpha to control the strength of the gravitational pull and the influence of initial conditions, respectively.
# The 'random_flight' operator involves scaling factor (scale), distribution type ('levy'), and beta parameter that affects how much the random flight influences the search direction. 
# These operators are configured with a selector named 'probabilistic', which means they will be applied probabilistically during each iteration, enhancing exploration in addition to exploitation.
# The benchmark function used is Rastrigin's function for optimization, defined over two dimensions (Rastrigin(2)), which helps in finding the optimal values for x and f based on their fitness in this function.