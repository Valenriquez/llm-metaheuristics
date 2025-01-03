# Name: Hybrid Metaheuristic for Global Optimization

# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.12668531445124898,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.004097018635755123,
            'alpha': 0.009068045819370141,
            'beta': 1.9971442629646592,
            'dt': 0.8842587616376874
        },
        'metropolis'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 0.7173036848463534
        },
        'probabilistic'
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
# 1. **Random Search**: Provides a simple and effective way to explore the solution space randomly.
# 2. **Central Force Dynamics**: Mimics the behavior of charged particles influenced by attractive and repulsive forces, which helps in navigating through complex landscapes.
# 3. **Differential Mutation**: Uses mutation operators based on differences between population members to generate new solutions, enhancing exploration.

# These operators together provide a balance between exploitation (local search) and exploration (global search), making the hybrid approach robust for global optimization problems like Rastrigin's function.