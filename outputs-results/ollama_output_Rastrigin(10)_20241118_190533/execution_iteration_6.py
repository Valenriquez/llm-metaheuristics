# Name: Hybrid Evolutionary Algorithm
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
    (
        'random_sample',
        {},
        'greedy'
    ),
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
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# - The hybrid evolutionary algorithm combines different search operators to leverage their strengths.
# - 'random_sample' provides a broad exploration of the solution space, helping to avoid local optima.
# - 'local_random_walk' fine-tunes solutions around promising areas, improving convergence rates.
# - 'swarm_dynamic' utilizes the collective behavior of particles to explore and exploit the search space efficiently.