# Name: Hybrid Metaheuristic Algorithm for Optimization

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
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand',
            'num_rands': 1,
            'factor': 1.0
        },
        'metropolis'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 100.0
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
# The hybrid metaheuristic algorithm combines several search operators to improve the exploration and exploitation capabilities of the optimization process. By using a combination of random search, central force dynamic, differential mutation, and firefly dynamic, the algorithm aims to balance between diversification and intensification. The 'greedy' selector is used for simple selection, while 'all', 'metropolis', and 'probabilistic' selectors enhance the robustness and efficiency of the search process. Running the metaheuristic multiple times allows us to evaluate its performance and gather statistical insights into the quality of the solutions obtained.