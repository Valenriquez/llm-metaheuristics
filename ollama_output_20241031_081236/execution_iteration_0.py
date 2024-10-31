# Name: Swarm Metaheuristic with Levy Distribution

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
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Swarm Metaheuristic with Levy Distribution is chosen because it is known to be effective for complex optimization problems. The Levy distribution is particularly suitable for this problem as it can generate diverse and exploratory solutions. The parameters are chosen based on the recommendations in parameters_to_take.txt. The probabilistic selector is used to ensure that solutions are selected based on their probability of being the best solution.