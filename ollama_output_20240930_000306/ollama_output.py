 # Name: Custom Genetic Algorithm with Gravitational Search and Random Flight
# Code:
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

# Define the search operators
heur = [
    (  # Genetic Mutation
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (  # Gravitational Search
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (  # Random Flight
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'gaussian',
            'beta': 1.5
        },
        'probabilistic'
    )
]

# Initialize and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and justification:
# This metaheuristic combines a genetic algorithm with mutation, gravitational search, and random flight to explore the solution space efficiently. 
# Genetic mutation introduces diversity into the population by randomly altering the gene values, while gravitational search mimics the physical force of gravity in optimizing positions. 
# Random flight is inspired by social forces that guide individuals towards beneficial regions, promoting global exploration. 
# By integrating these operators, we aim to balance local refinement and global exploration for better optimization outcomes across a range of benchmark functions.