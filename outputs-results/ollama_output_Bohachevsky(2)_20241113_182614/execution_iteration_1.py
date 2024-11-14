# Name: bohachevsky_metaheuristic
# Code:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)
prob = fun.get_formatted_problem()

spiral_dynamic_heur = [
    ('spiral_dynamic',
     {'radius': 0.1, 'angle': 22.5, 'sigma': 0.01},
     'random_sample'),
    ('spiral_dynamic',
     {'radius': 0.9, 'angle': 22.5, 'sigma': 0.01},
     'greedy')
]

swarm_dynamic_heur = [
    ('swarm_dynamic',
     {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'constriction', 'distribution': 'gaussian'},
     'random_sample'),
    ('swarm_dynamic',
     {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'constriction', 'distribution': 'uniform'},
     'greedy')
]

heur = spiral_dynamic_heur + swarm_dynamic_heur

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# Bohachevsky is a mathematical problem in optimization where we need to find the minimum function value.
# This metaheuristic will try two different approaches: spiral dynamic and swarm dynamic. Both are heuristic methods used for optimization.
# The goal of this task was to generate a more efficient metaheuristic algorithm by combining different heuristics with varying parameters.
# We should note that Bohachevsky's problem has very few local minima, but this does not affect our design because our objective is to improve the general efficiency of the metaheuristic. 
# A combination of more efficient methods may provide better performance.

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])