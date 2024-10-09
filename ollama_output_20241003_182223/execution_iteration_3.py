 # Name: GravitationalSearchOptimizer
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
# The algorithm is named GravitationalSearchOptimizer as it uses a gravitational search technique, which involves parameters such as gravity and alpha to guide the search process. 
# Two main operators are used: gravitational_search and random_flight. Gravitational search employs gravity and alpha for exploration and exploitation, while random flight includes scale, distribution (levy), and beta with a probabilistic selector. These operators help in exploring the solution space by balancing between local and global searches.
# The Sphere function is chosen as it is a common benchmark problem to test optimization algorithms due to its unimodal nature, making it easier to find the global minimum. 
# Parameters gravity (1.0) and alpha (0.02) are set according to typical settings for gravitational search parameters. Distribution 'levy' in random_flight helps in escaping local minima by following a levy flight distribution characteristic which is known to have long jumps and can explore distant regions of the search space. The beta parameter in random_flight controls the balance between exploration (beta > 1) and exploitation (beta < 1), where we set it to 1.5 for a bias towards exploration, as indicated by 'probabilistic' selector which uses probabilistic decisions based on fitness values.
# The selection of operators and their parameters is guided by literature recommendations for gravitational search and random flight mechanisms in optimization algorithms, ensuring a balance between convergence speed and accuracy through diverse exploration strategies.