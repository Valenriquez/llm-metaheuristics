 # Name: Multi-Strategy Evolutionary Algorithm (MSEA)
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operators and their parameters
heur = [
    ( # Search operator 1: Genetic Mutation
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    ( # Search operator 2: Gravitational Search
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'probabilistic'
    )
]

# Initialize the metaheuristic with the defined operators and problem
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Short explanation and justification:
# The Multi-Strategy Evolutionary Algorithm (MSEA) is designed to explore multiple search operators to enhance the exploration and exploitation capabilities of the algorithm. In this implementation, we include Genetic Mutation and Gravitational Search as our primary operators. 
# - Genetic Mutation with Gaussian distribution allows for both local and global explorative mutations based on Gaussian random numbers scaled by a factor, promoting diversity in the population while occasionally introducing significant changes.
# - Gravitational Search simulates the gravitational force among particles to adjust their positions according to predefined rules, which helps in converging towards promising regions of the search space. This is supported by the parameters gravity and alpha, controlling the strength and range of these forces within the population. 
# These operators are chosen based on their ability to complement each other's weaknesses and strengths in different problem-solving scenarios, providing a robust approach suitable for various benchmark functions through the iterations defined.