# Name: Multi-Objective Spiral Dynamic Metaheuristic (MOSDMH)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # Adjust this path based on your project structure
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(5)  # Replace with the selected problem and dimension.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'random_sample'
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
        'random_sample'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'random_sample'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)  # Adjust number of agents based on dimension
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep + 1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# MOSDMH combines Spiral Dynamic, Swarm Dynamic, and Local Random Walk operators to explore the solution space effectively. The spiral dynamic operator helps in navigating through the search space in a structured manner, while the swarm dynamic and local random walk operators ensure diverse exploration and exploitation. This combination aims to balance between exploration and exploitation, leading to better optimization results for complex problems.