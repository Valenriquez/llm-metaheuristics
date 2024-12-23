# Name: Adaptive Metaheuristic for Optimization

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
        'local_random_walk',
        {
            'probability': 0.7914567584180503,
            'scale': 1.48695186816469,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7656656910415142,
            'self_conf': 1.7433362198228672,
            'swarm_conf': 1.786089216430757,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.2027088808813204,
            'angle': 22.762100463147284,
            'sigma': 0.10704473738225637
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The chosen metaheuristic consists of a combination of local search operators like Local Random Walk and Spiral Dynamic,
# along with global search operators such as Swarm Dynamic. The inclusion of both types helps in effectively exploring the
# solution space, balancing exploration and exploitation. The Metropolis selector is used for random sampling to ensure diversity.
# This approach aims to find high-quality solutions within a reasonable number of iterations.

# If any part of the data is incomplete or unavailable, state explicitly: "Information not available."