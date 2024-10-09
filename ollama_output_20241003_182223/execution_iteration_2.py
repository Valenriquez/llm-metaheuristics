 # Name: GravitationalSearchAndRandomFlightMetaheuristic
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
# This metaheuristic combines two search operators: gravitational_search and random_flight. The gravitational_search uses a gravity parameter of 1.0 and an alpha value of 0.02, while the selector for this operator is set to 'metropolis'. The random_flight operator has a scale of 1.0, a distribution type set to 'levy', and a beta value of 1.5, with its selector also set to 'probabilistic'. These parameters are chosen based on the information provided in parameters_to_take.txt, ensuring that only valid operators and parameters are used. The combination of gravitational search's exploitative power and random flight's exploratory nature should lead to a balanced optimization approach, potentially improving the convergence speed and accuracy of the algorithm for solving the benchmark function.