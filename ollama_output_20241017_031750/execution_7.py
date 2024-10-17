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
# This metaheuristic combines the Gravitational Search Algorithm (GSA) with a Random Flight mechanism. 
# The GSA is designed to mimic gravitational interactions among particles in a system, where 'gravity' represents the strength of these interactions and 'alpha' controls the exponential decay rate. 
# In contrast, the random flight operator introduces stochastic elements by scaling the search space ('scale') with either a levy or uniform distribution, while 'beta' influences the probabilistic nature of this movement.
# The combination allows for both systematic exploration (GSA) and adaptive randomness (random flight), which is expected to enhance the algorithm's ability to escape local minima and explore broader regions of the search space. 
# The use of 'probabilistic' selector ensures that random flight decisions are made with consideration of probability, promoting diverse search patterns across iterations.