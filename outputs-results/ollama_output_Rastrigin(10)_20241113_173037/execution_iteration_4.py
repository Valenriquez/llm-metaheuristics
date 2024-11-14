# name: rastrigumeta
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
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.7,
            'scale': 1.2,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 25.0,
            'sigma': 0.15
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# Rastrigin is a multimodal function with 10 dimensions. The goal of this metaheuristic is to find the minimum fitness value.
# In each iteration, we use a combination of local_random_walk and spiral_dynamic operators. 
# The selection strategy uses greedy and metropolis strategies.

# Run the metaheuristic with the same problem 30 times
fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])