# Name: Swarm Metaheuristic with Inertial Version and Gaussian Distribution

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)   
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
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
# We choose the swarm metaheuristic with inertial version and Gaussian distribution
# because it has been shown to be effective in optimizing complex problems.
# The inertial version promotes exploration and exploitation of the search space,
# while the Gaussian distribution encourages diversification of the swarm's positions.
# By combining these features, we believe this metaheuristic can effectively find
# optimal solutions for the Rastrigin(2) optimization problem.