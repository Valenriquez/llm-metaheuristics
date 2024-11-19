# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)
prob = fun.get_formatted_problem()

heur = [
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
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
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
# HybridMetaheuristic combines two search operators: Swarm Dynamic and Spiral Dynamic. The Swarm Dynamic operator is initialized with parameters that promote exploration, while the Spiral Dynamic operator encourages exploitation by focusing on potential solutions in a spiral pattern. This hybrid approach aims to balance exploration and exploitation effectively, leading to potentially better solutions for complex optimization problems like Rastrigin's function.
# The use of 'all' selector for Swarm Dynamic ensures thorough exploration across all feasible regions, while the 'probabilistic' selector for Spiral Dynamic allows for more focused exploitation around promising areas. This combination is particularly suitable for high-dimensional and multimodal problems where both global search and local refinement are necessary.