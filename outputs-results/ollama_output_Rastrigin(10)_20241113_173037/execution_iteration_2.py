# Name: spirandynametaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)
prob = fun.get_formatted_problem()

heur = [
    ('spiral_dynamic', {
        'radius': 1.0,
        'angle': 22.5,
        'sigma': 0.1,
    }, 'greedy'),
    ('swarm_dynamic', {
        'factor': 0.7,
        'self_conf': 2.54,
        'swarm_conf': 2.56,
        'version': 'constriction',
        'distribution': 'uniform',
    }, 'metropolis')
]

def spiral_dynamic(x):
    # Implement Spiral Dynamic algorithm
    return x

def swarm_dynamic(x):
    # Implement Swarm Dynamic algorithm
    return x

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The generated metaheuristic combines two different search operators: Spiral Dynamic and Swarm Dynamic.
# Each operator has its own parameters that are used to influence the behavior of the algorithm.
# The 'greedy' selector is used for the first operator, which selects the best solution found so far.
# The 'metropolis' selector is used for the second operator, which uses a more complex selection process based on the Metropolis algorithm.

# Run the metaheuristic with the same problem 30 times to get a better estimate of its performance
fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])