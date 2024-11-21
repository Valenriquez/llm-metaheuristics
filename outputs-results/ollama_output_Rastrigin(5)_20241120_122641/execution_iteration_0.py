# Name: Custom Metaheuristic for Rastrigin Function
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial'
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
# This metaheuristic uses two search operators: 'swarm_dynamic' and 'spiral_dynamic'. 
# The 'swarm_dynamic' operator is designed for multi-dimensional problems and can adapt its behavior based on the self-confidence (self_conf) and swarm confidence (swarm_conf). The 'inertial' version of this operator ensures a smoother exploration of the search space.
# The 'spiral_dynamic' operator helps in fine-tuning the solution by exploring the vicinity of the current best solutions. It uses parameters like radius, angle, and sigma to control the spiral's path around the optimal point.
# The use of both operators together allows for a comprehensive exploration and exploitation strategy that is well-suited for the Rastrigin function, which has many local minima and a large search space.