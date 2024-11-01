# Name: SwarmMetaheuristic
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
    (
        'swarm_dynamic',
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
# We choose the swarm_dynamic operator with the levy distribution, inertia version, and probabilistic selector.
# These settings have been shown to be effective in solving optimization problems using metaheuristics.
# The factor parameter controls the strength of the swarm, while the self_conf and swarm_conf parameters determine the confidence of the individual and swarm, respectively.
# The probabilistic selector ensures that the best solutions are more likely to be selected, while still allowing for exploration of other solutions.