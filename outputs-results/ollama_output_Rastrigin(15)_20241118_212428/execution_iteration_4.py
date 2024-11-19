# Name: Hybrid Metaheuristic with Spiral and Swarm Dynamics  
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
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
    ('swarm_dynamic',
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
# The Hybrid Metaheuristic with Spiral and Swarm Dynamics combines two powerful search strategies: Spiral Dynamic and Swarm Dynamic. 
# The Spiral Dynamic helps in exploring the solution space efficiently by following a spiral pattern, ensuring that all promising regions are covered.
# The Swarm Dynamic, on the other hand, mimics the behavior of social insects to find the best solution quickly. By combining these two approaches, we aim to leverage the strengths of both for a robust optimization process. 
# The use of the 'all' selector ensures that each agent can explore and exploit the search space comprehensively.