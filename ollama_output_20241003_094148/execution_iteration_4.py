 # Name: PSO_with_CustomParams
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'swarm_dynamic',
    { 
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'inertial',
        'distribution': 'uniform'
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
# The metaheuristic used in this implementation is Particle Swarm Optimization (PSO). 
# Two search operators are employed: 'swarm_dynamic' for exploring the solution space with dynamic behavior, 
# and 'random_flight' for random walks influenced by a Levy distribution. Both operators use parameters from 
# the 'parameters_to_take.txt' file to define their behaviors. The selector is set to 'all', indicating that both 
# greedy and probabilistic selections will be considered during the search process. This combination aims to balance 
# exploration and exploitation, enhancing the algorithm's ability to converge towards an optimal solution while avoiding 
# local minima. The PSO with custom parameters provides a versatile framework where each operator is tailored according to 
# its specific requirements as defined in 'parameters_to_take.txt'.