# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # Selected problem with dimension 5.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_flight',
        {
            'scale': 1.2,
            'distribution': 'gaussian',
            'beta': 1.5
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 30,
            'sigma': 0.05
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.4,
            'swarm_conf': 2.4,
            'version': 'constriction',
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
# HybridMetaheuristic combines the strengths of three different search operators: Random Flight, Spiral Dynamic, and Swarm Dynamic. 
# Random Flight helps in exploring the solution space extensively, Spiral Dynamic ensures a gradual convergence towards an optimal solution, 
# and Swarm Dynamic facilitates global exploration while leveraging the collective intelligence of multiple agents.
# This hybrid approach is expected to perform well on the Rastrigin function by balancing exploration and exploitation effectively.