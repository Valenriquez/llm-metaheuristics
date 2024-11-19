# Name: Hybrid Metaheuristic for Rastrigin Function Optimization
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 2.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8,
            'self_conf': 3.0,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 18.0,
            'sigma': 0.2
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The hybrid metaheuristic combines three different optimization techniques to maximize the efficiency of finding the global minimum in the Rastrigin function. 
# - 'local_random_walk' helps escape local minima.
# - 'swarm_dynamic' utilizes swarm intelligence for searching the solution space thoroughly.
# - 'spiral_dynamic' is employed to fine-tune and polish the solution near the optimal region.