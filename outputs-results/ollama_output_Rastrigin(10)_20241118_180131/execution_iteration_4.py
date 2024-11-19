# Name: Hybrid Metaheuristic with Adaptive Operators
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
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
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

# Short explanation and justification:
# This hybrid metaheuristic combines two operators: 'local_random_walk' and 'spiral_dynamic'. Both are used together to explore the search space more effectively. The 'local_random_walk' operator is effective for fine-grained exploration, while the 'spiral_dynamic' operator helps in larger-scale exploration. Using both operators together allows for a balanced search strategy that can potentially find better solutions faster and with higher quality.
# The selection of 'gaussian' distribution for the 'local_random_walk' operator ensures smooth and controlled movements around the current solution, making it suitable for local refinement. The 'spiral_dynamic' operator's parameters are set to facilitate efficient exploration without getting stuck in suboptimal regions.
# Using 'all' as the selector for both operators ensures that they are applied evenly throughout the entire search process, giving each operator an opportunity to contribute to the solution-finding effort.