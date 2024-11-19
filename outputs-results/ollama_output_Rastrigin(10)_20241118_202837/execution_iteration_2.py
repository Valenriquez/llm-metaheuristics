# Name: Adaptive Hybrid Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'  # Using 'all' selector since the problem is multi-dimensional
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'  # Using 'all' selector since the problem is multi-dimensional
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'  # Using 'all' selector since the problem is multi-dimensional
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Adaptive Hybrid Metaheuristic combines the strengths of local search with swarm intelligence and spiral dynamics to efficiently explore the solution space. Each operator is used in parallel, allowing for a more thorough exploration of the multi-dimensional Rastrigin function's landscape.
#
# 'local_random_walk' helps to refine solutions by making small, probabilistic moves towards potentially better solutions.
#
# 'swarm_dynamic' uses particle swarm optimization principles, where particles (solutions) are guided by both their personal best and the global best positions to explore the solution space collectively.
#
# 'spiral_dynamic' employs a spiral trajectory for exploration, which can help in escaping local optima and exploring new regions of the search space.
#
# By combining these operators in an adaptive hybrid manner, we aim to achieve a balance between exploitation and exploration, leading to more efficient and effective optimization results.