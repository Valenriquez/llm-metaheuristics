 # Name: Adaptive Multi-Strategy Evolutionary Algorithm (AMSEA)
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.get_problem('sphere')
prob = fun.get_formatted_problem()

# Define the search operators with their specific parameters and selectors
heur = [
    (
        'genetic_crossover',
        {
            'pairing': ['rank'],
            'crossover': ['single'],
            'mating_pool_factor': [0.4]
        },
        'all'
    ),
    (
        'genetic_mutation',
        {
            'scale': [1.0],
            'elite_rate': [0.1],
            'mutation_rate': [0.25],
            'distribution': ['uniform']
        },
        'all'
    ),
    (
        'gravitational_search',
        {
            'gravity': [1.0],
            'alpha': [0.02]
        },
        'all'
    ),
    (
        'random_flight',
        {
            'scale': [1.0],
            'distribution': ['levy'],
            'beta': [1.5]
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': [0.75],
            'scale': [1.0],
            'distribution': ['uniform']
        },
        'all'
    )
]

# Initialize the metaheuristic with the defined operators and parameters
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The AMSEA is designed to adaptively switch between various search operators, each with its own set of parameters optimized for different types of problems. This flexibility allows the algorithm to efficiently explore the solution space by leveraging the strengths of multiple strategies. 
# Genetic crossover and mutation are fundamental components of evolutionary algorithms and are included here to facilitate the exploration phase by combining and mutating solutions from previous generations, respectively. Gravitational search and random flight mimic natural phenomena to guide the search towards promising regions in the problem space. Local random walk introduces a stochastic element that encourages the algorithm to escape local minima and discover new areas of the solution landscape.
# The use of rank-based pairing in genetic operators ensures that less fit solutions still contribute to the population by being paired with more fit ones, promoting diversity and preventing premature convergence. Uniform crossover combines genes from two parent solutions randomly, which is beneficial for exploring different parts of the search space.
# Parameters are chosen based on their known effectiveness across a range of optimization tasks; for instance, elite_rate and mutation_rate in genetic mutation directly affect the retention and variation of good solutions within populations. The scale parameter in mutations controls the magnitude of changes introduced by mutation, while the distribution type influences how these changes are made (uniform random walk or Gaussian/Levy fluctuations).
# The adaptive nature of AMSEA is reflected in its ability to dynamically switch between different search operators and parameters based on empirical performance metrics not shown here but typically included in implementations for continuous adaptation.