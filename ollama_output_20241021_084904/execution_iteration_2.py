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
    'greedy' or 'all' or 'metropolis' or 'probabilistic'
    ),
    (  
    'gravitational_search',
    {
        'gravity': 1.0,
        'alpha': 0.02
    },
    'greedy' or 'all' or 'metropolis' or 'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm, which is characterized by its gravity parameter affecting the search direction and alpha controlling the step size. The Rastrigin function is chosen as the benchmark problem due to its multimodal nature, requiring multiple local searches to find all modes. Two instances of the gravitational search are employed with identical parameters for exploration across the solution space. The selectors used (greedy, all, metropolis, probabilistic) allow flexibility in how the algorithm explores the solution space, enabling a balance between exploitation and exploration as per the parameter settings.
#