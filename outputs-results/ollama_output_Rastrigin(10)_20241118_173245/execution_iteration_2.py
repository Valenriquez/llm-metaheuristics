# Name: Hybrid Metaheuristic with Adaptive Operators
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# 1. **Random Sample**: This operator samples solutions uniformly at random from the search space. It is useful for covering a broad area of the solution space.
# 2. **Local Random Walk**: This operator performs small, random steps in the neighborhood of the current solution. It helps in fine-tuning the solution and avoiding premature convergence.
# 3. **Swarm Dynamic**: Inspired by the behavior of social insects, this operator uses a swarm of particles that move towards the best solutions found so far. It is particularly effective for multi-dimensional problems.
#
# Together, these operators provide a balance between exploration and exploitation, making the metaheuristic more robust and efficient in solving complex optimization problems like the Rastrigin function.