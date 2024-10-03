 # Name: GravitationalSearchAlgorithmMetaheuristic
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed based on the Gravitational Search Algorithm (GSA), which mimics the gravitational force among masses to explore and optimize a search space. The algorithm uses two main parameters: 'gravity' and 'alpha'. 'Gravity' represents the strength of the gravitational force, controlling the exploration versus exploitation trade-off. A higher value of gravity promotes more global exploration by allowing larger step sizes, while a lower value directs the search towards local optima through smaller steps. 'Alpha', on the other hand, is a scaling factor that influences the contribution of individual solutions to the overall search process, affecting both convergence speed and accuracy. The selector 'all' indicates that this operator will be applied to all individuals in the population at each iteration, allowing for simultaneous global and local explorations.
