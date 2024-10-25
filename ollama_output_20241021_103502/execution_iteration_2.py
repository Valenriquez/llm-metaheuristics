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
# This metaheuristic combines the Gravitational Search algorithm with the Random Flight operator for exploration. The Gravitational Search, parameterized with a gravity of 1.0 and an alpha value of 0.02, is designed to simulate the gravitational force between particles in space, guiding them towards areas of lower energy (or better fitness). This helps in convergence towards the optimal solution.
# The Random Flight operator introduces stochasticity by scaling the step size according to a levy distribution or uniform distribution, depending on the beta parameter. This allows for both exploration and exploitation, as the algorithm can either take large steps when close to the optimum or small random steps to avoid getting stuck in local minima.
# The selector 'probabilistic' is chosen for the Random Flight operator because it introduces randomness into the search process with a probability-based decision mechanism, which aligns well with the probabilistic nature of this operator. Both operators are applied in each iteration to leverage their respective strengths for effective optimization.