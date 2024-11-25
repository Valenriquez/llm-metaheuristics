# Name: Spiral-Swarm Hybrid Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.5169751902406856,
            'angle': 1.3732174916203164,
            'sigma': 0.12132056730164364
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.2920579376017425,
            'self_conf': 2.54,
            'swarm_conf': 2.54,
            'version': 'inertial'
        },
        'random_walk'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines Spiral Dynamic (spiral_dynamic) and Swarm Dynamic (swarm_dynamic) operators to explore the search space.
# The Spiral Dynamic operator is used for initial exploration with a spiral path, while the Swarm Dynamic operator refines the search using particle swarm optimization principles.
# Both operators are selected with specific parameters tailored for the Rastrigin function in 13 dimensions.