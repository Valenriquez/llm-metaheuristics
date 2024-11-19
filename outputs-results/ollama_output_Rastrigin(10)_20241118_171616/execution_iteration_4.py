# Name: Hybrid Metaheuristic for Rastrigin Function Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': "gaussian"
        },
        'all'
    ),
    (
        # Search operator 2: Spiral Dynamic
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

# Short explanation and justification:
# The Hybrid Metaheuristic for Rastrigin Function Optimization combines the Local Random Walk (LRW) and Spiral Dynamic (SD) operators to explore the search space efficiently.
# LRW helps in fine-tuning the solution by making small, controlled changes with a certain probability, which is beneficial around local minima.
# SD promotes global exploration by moving along spiral paths, which can help escape from local minima and potentially find better solutions.
# The use of 'all' selector ensures that both operators are considered at every step, allowing for a balance between exploration and exploitation.