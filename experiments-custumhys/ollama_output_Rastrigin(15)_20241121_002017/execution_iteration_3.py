# Name: Particle Swarm Optimization with Spiral Dynamic and Gaussian Distribution

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # Example problem: Rastrigin function with dimension 15
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.6318196136628489,
            'angle': 2.067774088773747,
            'sigma': 0.11226822266391662
        },
        'probabilistic'
    ),
    (  # Search operator 2: Swarm Dynamic with Gaussian Distribution
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'greedy'
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
# The metaheuristic combines Spiral Dynamic with Gaussian Distribution for better exploration and exploitation.
# Spiral Dynamic helps in navigating through the solution space more effectively by using a spiral path.
# Swarm Dynamic with Gaussian Distribution ensures efficient convergence towards the global optimum.

# Error Fix: Corrected the syntax error and formatting issues.