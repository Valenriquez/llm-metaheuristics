# Name: Spiral Dynamic Swarm Optimization (SDSO)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)  # This is the selected problem.
prob = fun.get_formatted_problem()

num_agents = 15 * prob['dimensions']

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines the Spiral Dynamic operator with the Swarm Dynamic operator to explore the search space efficiently. The Spiral Dynamic operator helps in moving around the global optimum, while the Swarm Dynamic operator ensures a good balance between exploration and exploitation by adjusting the cognitive and social parameters.