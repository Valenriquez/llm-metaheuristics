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
# This metaheuristic combines the Gravitational Search algorithm with the Random Flight operator for exploration. The Gravitational Search, characterized by its 'gravity' parameter set to 1.0 and 'alpha' set to 0.02, is designed to simulate gravitational forces acting on particles within a search space, promoting convergence towards better solutions. On the other hand, the Random Flight operator with parameters 'scale' at 1.0, distribution type 'levy', and 'beta' of 1.5 introduces stochastic elements into the search, aiding in diversification and exploration of the solution space. The selector for both operators is set to 'probabilistic', which uses a probability-based selection mechanism that favors better solutions with a certain probability, enhancing the overall performance by balancing between exploitation and exploration. This hybrid approach aims to leverage the strengths of both algorithms to efficiently converge towards an optimal solution while maintaining diversity in the search space.