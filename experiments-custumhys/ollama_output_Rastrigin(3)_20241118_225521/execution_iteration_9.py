# Name: Hybrid Evolutionary Algorithm with Spiral Dynamic and Swarm Dynamics

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
    ('spiral_dynamic', 
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
# The hybrid evolutionary algorithm combines the Spiral Dynamic and Swarm Dynamics operators to improve the exploration and exploitation capabilities of the search space. 
# The Spiral Dynamic operator helps in finding new regions by exploring around promising solutions, while the Swarm Dynamics operator enhances the exploitation of these regions by utilizing a group of agents that communicate and share information. 
# Both selectors are used to ensure that each operator has an equal chance of being selected at every iteration, promoting a balanced search process. This combination is particularly useful for problems with complex landscapes and multiple local optima.