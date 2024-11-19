# Name: Custom Metaheuristic for Rastrigin Function
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # Adjust this line accordingly if your project structure changes.
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
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

# Short explanation and justification:
# The metaheuristic is designed to solve the Rastrigin function in 10 dimensions. It uses a combination of two search operators: 'local_random_walk' and 'spiral_dynamic'. The 'local_random_walk' operator helps in exploring the solution space by taking random steps with varying probabilities, while the 'spiral_dynamic' operator allows for more systematic exploration using a spiral pattern that converges towards the optimum. The combination aims to balance exploration and exploitation effectively.