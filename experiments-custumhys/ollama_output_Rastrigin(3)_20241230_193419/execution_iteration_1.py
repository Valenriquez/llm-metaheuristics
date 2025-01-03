# Name: HybridMetaheuristic

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
            'scale': 0.3428357115298649,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'gaussian',
            'alpha': 0.680145335633586,
            'beta': 2.4816176252379414,
            'gamma': 5.648288434162795
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.002589250765573318,
            'alpha': 0.015,
            'beta': 1.8,
            'dt': 0.9954686214379734
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True

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
# 'random_search' is used for initial exploration with low scale to cover a broad area.
# 'firefly_dynamic' helps in refining the solution by mimicking the behavior of fireflies, enhancing local convergence.
# 'central_force_dynamic' introduces global search capabilities, guiding the population towards better solutions.
# The combination of these operators allows the algorithm to balance between exploration and exploitation, potentially leading to better optimization results.