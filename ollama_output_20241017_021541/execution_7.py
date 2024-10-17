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
    'metropolis'
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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with two main operators: gravitational_search and random_flight. 
# The gravitational_search operator uses a gravity parameter of 1.0 and an alpha value of 0.02, while the selector for this operator is set to 'metropolis'. 
# The random_flight operator employs a scale of 1.0 with a distribution type of 'levy' and a beta value of 1.5, using 'probabilistic' selection. 
# These parameters are chosen based on the specifications provided in parameters_to_take.txt to ensure effective exploration and exploitation within the optimization process. The use of both operators allows for a comprehensive search across the problem space, leveraging the strengths of gravitational and random flight dynamics to converge towards an optimal solution.