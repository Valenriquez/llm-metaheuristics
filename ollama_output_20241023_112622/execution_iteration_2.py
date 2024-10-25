 # Name: GravitationalSearchMetaheuristic
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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm, which is inspired by the laws of gravity and physics. The first operator in the heuristic uses 'gravitational_search' with parameters 'gravity' set to 1.0 and 'alpha' to 0.02. This operator will be applied to all candidates during the search process.
# The second operator utilizes 'random_flight' with 'scale' set to 1.0, distribution type 'levy', and a beta value of 1.5. It is selected as probabilistic because it introduces randomness into the search based on certain probability distributions. This helps in exploring different regions of the solution space during optimization.
# The GravitationalSearchMetaheuristic combines both deterministic gravitational attraction for convergence and stochastic random flight for exploration, aiming to balance between exploitation and exploration effectively according to the problem requirements set by 'gravitational_search' and 'random_flight' parameters respectively.