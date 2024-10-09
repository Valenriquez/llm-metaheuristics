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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with two main search operators. 
# The first operator, gravitational_search, utilizes a gravity parameter of 1.0 and an alpha value of 0.02. This operator uses all possible selections ('all') during its iterations.
# The second operator is random_flight, which scales at 1.0 with a distribution set to 'levy'. It selects options probabilistically ('probabilistic').
# These operators are chosen based on the parameters specified in the parameters_to_take.txt file, ensuring that only allowed operators and parameters are utilized. The combination of gravitational search for exploration and random flight for diversification aims to balance between local and global search capabilities, which is typical for metaheuristic approaches in optimization problems.