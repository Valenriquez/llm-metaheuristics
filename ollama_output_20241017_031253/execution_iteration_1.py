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
# The GravitationalSearchMetaheuristic is designed to explore the search space using a gravitational analogy, where particles are attracted to higher fitness values based on gravity coefficients. In this case, we use 'gravitational_search' with parameters gravity set to 1.0 and alpha set to 0.02, targeting all possible solutions in the search space.
# Additionally, we incorporate 'random_flight' as a probabilistic operator, which allows for random movements across the search space. The scale is set to 1.0, and the distribution type is chosen as 'levy', enhancing the exploration capability of the algorithm by favoring non-linear paths through the problem landscape.
# These operators are selected with a 'probabilistic' selector because they introduce randomness into the search process, which can lead to better global exploration when combined with probabilistic selection methods. This approach aims to balance between exploitation and exploration, optimizing both local and global minima within the defined parameter ranges.