# Name: Enhanced PSO with Local Random Walk

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# Enhanced PSO with Local Random Walk combines the global search capability of Particle Swarm Optimization (PSO) with a local refinement technique using Local Random Walk.
# The swarm_dynamic operator is used to explore the solution space globally, leveraging PSO's collective intelligence to converge towards promising regions.
# The local_random_walk operator is introduced to perform fine-grained exploration around the best-known solutions, enhancing convergence and potentially improving accuracy for complex benchmark functions like Rastrigin.
# This hybrid approach leverages both the broad search ability of PSO and the detailed exploration capabilities of Local Random Walk, leading to more efficient optimization and potentially better solution quality.