 # Name: GravitationalSearchAndRandomFlightMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(22)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'metropolis'
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
# This metaheuristic combines the gravitational search and random flight operators to explore potential solutions in a diverse manner. The Gravitational Search operator uses parameters 'gravity' (1.0) and 'alpha' (0.02), aiming to simulate the force of gravity for attracting particles towards higher-quality solutions. The Random Flight operator, with its 'scale' set to 1.0, employs different distributions ('levy', 'uniform', or 'gaussian') depending on the scenario, allowing it to escape local minima and explore broader areas of the solution space. Both operators are configured with a probabilistic selector to ensure they contribute effectively during each iteration. This combination is expected to yield better exploration of the search space and potentially find more optimal solutions compared to using a single operator.