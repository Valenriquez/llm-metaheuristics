# Name: Hybrid Metaheuristic for Benchmark Function Optimization
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # Example with Rastrigin function and 10 dimensions.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Gravitational Search Algorithm
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'all'
    ),
    (  # Search operator 3: Genetic Algorithm with Crossover and Mutation
        'genetic_crossover',
        {
            'pairing': 'cost',
            'crossover': 'uniform',
            'mating_pool_factor': 0.4
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This hybrid metaheuristic combines three search operators: Gravitational Search Algorithm (GSA), Random Flight, Genetic Crossover, and Genetic Mutation. GSA helps in exploring the solution space globally, Random Flight provides local refinement using Levy flights, while Genetic Algorithms ensure diversity and exploitation of promising regions. The hybrid approach leverages the strengths of each operator to efficiently solve high-dimensional optimization problems like Rastrigin.