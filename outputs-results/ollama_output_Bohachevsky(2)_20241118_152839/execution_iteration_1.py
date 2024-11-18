# Name: Multi-Objective Hybrid Metaheuristic using Genetic Algorithm and Gravitational Search

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.benchmark_function_1(dimension=3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'genetic_crossover',
        {
            'pairing': 'rank',
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
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=50)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic combines the strengths of Genetic Algorithm (GA) for exploring the search space and finding good solutions,
# while Gravitational Search Algorithm (GSA) guides the GA by utilizing its global searching ability. 
# We use 'rank' pairing strategy for the GA to ensure a fair selection process, 'uniform' crossover operation to maintain diversity in the population,
# and Gaussian mutation rate for better exploration capability.
# GSA employs gravitational forces based on fitness values that can help avoid premature convergence and enhance search efficiency.
# The 'greedy', 'all', and 'metropolis' selectors represent our choices of algorithm selection strategies within each operator, 
# aiming to balance solution quality and computational time. This combination is expected to be effective for the given problem dimensions.