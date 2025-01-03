# Name: Multi-Operator Metaheuristic (MOM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.9168011915931125,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0013946596298128264,
            'alpha': 0.02764699951701834,
            'beta': 1.7164984779678796,
            'dt': 0.253921681583559
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 2,
            'factor': 0.9079675721674361
        },
        'probabilistic'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 1.7326095031634556
        },
        'metropolis'
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
    print('rep = {}, x_best = {}, f_best = {}'.format(rep + 1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

# Short explanation and justification:
# This metaheuristic combines multiple operators to explore the solution space efficiently. The `random_search` operator helps in exploring diverse solutions, while `central_force_dynamic`, `differential_mutation`, and `firefly_dynamic` leverage physical-based dynamics to guide the search process. This combination aims to balance exploration and exploitation, leading to better convergence and a higher likelihood of finding the global optimum.