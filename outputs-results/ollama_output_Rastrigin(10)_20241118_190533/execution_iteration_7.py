# Name: Multi-Operator Hybrid Metaheuristic (MOHM)
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
            'scale': 2.0,
            'distribution': "uniform"
        },
        'all'
    ),
    (
        # Search operator 2: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 18.0,
            'sigma': 0.5
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Multi-Operator Hybrid Metaheuristic (MOHM) combines two search operators: Local Random Walk and Spiral Dynamic. 
# This hybrid approach aims to balance exploration and exploitation in the search space, potentially leading to better convergence and solution quality for the Rastrigin function.
#
# - **Local Random Walk**: This operator helps to explore the neighborhood of the current best solution, which is crucial for finding local minima. The higher probability and scale values enhance its exploration capabilities.
# - **Spiral Dynamic**: This operator follows a spiral trajectory to efficiently cover the search space, ensuring that important regions are visited thoroughly. Adjusting the radius, angle, and sigma parameters allows it to adapt to different problem structures.
#
# Together, these operators provide a robust framework for optimizing the Rastrigin function, balancing the need for thorough exploration and effective exploitation of promising areas in the search space.