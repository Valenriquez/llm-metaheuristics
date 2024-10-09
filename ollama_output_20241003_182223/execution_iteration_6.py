 # Name: GravitationalSearchMetaheuristic
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
    'all'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': "levy",
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
# The algorithm is named GravitationalSearchMetaheuristic as it combines the gravitational search operator with a random flight operator. 
# The gravitational search operator uses parameters gravity (1.0) and alpha (0.02), which control the strength of the gravitational force and the scaling factor, respectively. This helps in exploring the solution space.
# The random flight operator involves scale (1.0), distribution ("levy"), and beta (1.5). The levy distribution is chosen for its ability to explore different directions in the search space, while the scale and beta parameters control the exploration intensity.
# Both operators are used with a probabilistic selector, allowing for a balance between deterministic and stochastic exploration, which can lead to better convergence towards the global optimum.
# This setup ensures that we leverage both gravitational attraction and random jumps to effectively navigate through the problem space, using parameters directly from the provided configuration file.