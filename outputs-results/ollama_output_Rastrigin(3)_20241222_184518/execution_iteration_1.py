# Name: HybridMetaheuristic

# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',  # Search operator 2: Central Force Dynamic
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'probabilistic'
    ),
    (
        'differential_mutation',  # Search operator 3: Differential Mutation
        {
            'expression': 'rand-to-best',
            'num_rands': 1,
            'factor': 1.0
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

# Short explanation and justification:
# The HybridMetaheuristic combines three different search operators to explore the solution space more effectively. 
# Random Search helps in exploring uncharted territories by making random moves.
# Central Force Dynamic simulates the gravitational effect among particles, guiding them towards better solutions.
# Differential Mutation adapts the mutation strategy based on the best individuals in the population, enhancing convergence.
# The combination of these operators allows for a robust exploration-exploitation balance, improving the chances of finding high-quality solutions.