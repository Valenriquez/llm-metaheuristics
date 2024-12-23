# Name: Metaheuristic with Hybrid Operators and Selectors for Rastrigin Function Optimization
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'central_force_dynamic',
        {
            'gravity': 0.01,
            'alpha': 0.02,
            'beta': 1.5,
            'dt': 1.0
        },
        'greedy'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 1,
            'factor': 1.0
        },
        'all'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'gaussian',
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 50.0
        },
        'metropolis'
    ),
    (
        'gravitational_search',
        {
            'gravity': 0.8,
            'alpha': 0.03
        },
        'probabilistic'
    ),
    (
        'random_flight',
        {
            'scale': 1.2,
            'distribution': 'uniform',
            'beta': 1.5
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.7,
            'scale': 0.9,
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.85,
            'angle': 25.0,
            'sigma': 0.1
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.75,
            'self_conf': 2.4,
            'swarm_conf': 2.58,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'probabilistic'
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
# This metaheuristic combines multiple search operators (central_force_dynamic, differential_mutation, firefly_dynamic, gravitational_search, random_flight, local_random_walk, spiral_dynamic, swarm_dynamic) with various selection strategies (greedy, metropolis, probabilistic). Each operator is designed to explore different aspects of the solution space, while the selection strategy helps in efficiently converging to optimal solutions. This hybrid approach aims to balance exploration and exploitation, making it effective for optimizing complex problems like the Rastrigin function.