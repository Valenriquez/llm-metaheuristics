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
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8133438701103461,
            'self_conf': 2.9223780045285754,
            'swarm_conf': 2.0272879413297074,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.15315251719164122,
            'angle': 24.55423676380834,
            'sigma': 0.07810075731963893
        },
        'probabilistic'
    ),
    (
        'random_sample',
        {},
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
# The HybridMetaheuristic combines multiple search operators to leverage their strengths in solving the Rastrigin function. 
# 'random_sample' provides initial solutions randomly, while 'swarm_dynamic' simulates a swarm of particles with cognitive and social components.
# 'spiral_dynamic' uses a spiral movement strategy for exploration. The metropolis selector is used for local search around the best solution found.
# This hybrid approach aims to balance exploration and exploitation, potentially leading to more efficient convergence on the Rastrigin function.