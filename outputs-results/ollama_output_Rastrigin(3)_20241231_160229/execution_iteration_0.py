# Name: Hybrid Metaheuristic for Rastrigin Function

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8892166349315961,
            'scale': 0.6539455328983005,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5145600947044427,
            'self_conf': 2.297541183606822,
            'swarm_conf': 2.9579903845003264,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5270334350533005,
            'angle': 20.268988614118374,
            'sigma': 0.10211011920853721
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

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
# This hybrid metaheuristic combines multiple search operators to explore the Rastrigin function landscape more effectively.
# 'random_sample' provides a global perspective by randomly sampling points.
# 'local_random_walk' allows for fine-grained exploration around current solutions.
# 'swarm_dynamic' simulates a swarm behavior, utilizing social interactions to enhance search efficiency.
# 'spiral_dynamic' guides the search in a spiral manner, leveraging both local and global information.
# By integrating these operators, the metaheuristic aims to balance exploration and exploitation, leading to better convergence and higher quality solutions.