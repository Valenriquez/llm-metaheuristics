# Name: Swarm Metaheuristic with Inertial Version and Levy Distribution

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
    ('swarm_dynamic',
     {'factor': 0.7,
      'self_conf': 2.54,
      'swarm_conf': 2.56,
      'version': 'inertial',
      'distribution': 'levy'},
     'probabilistic'),
    ('random_sample',
     {},
     'greedy')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the swarm dynamic operator with the random sample operator.
# The swarm dynamic operator promotes exploration by adjusting the swarm's configuration.
# The random sample operator provides diversification by selecting new candidate solutions.
# The levy distribution provides a probability distribution for selecting new candidate solutions based on their proximity to the current best solution.