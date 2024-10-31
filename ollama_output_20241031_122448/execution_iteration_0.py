## Name: Swarm-Based Metaheuristic with Levy Distribution

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
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'levy'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses the swarm_dynamic operator with the levy distribution. 
# The levy distribution provides a strong exploration capability, which is crucial for escaping local optima in complex optimization problems like the bf.Rastrigin(2) function. 
# The probabilistic selector is chosen to guide the swarm towards promising candidate solutions with a high probability.
# The inertial version of the swarm dynamics is used to simulate the social interactions and knowledge sharing within the swarm, leading to faster convergence and improved performance.