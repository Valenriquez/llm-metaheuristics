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
# The SwarmMetaheuristic uses the swarm_dynamic operator with the levy distribution and probabilistic selector to optimize the Rastrigin(2) function.
# The swarm_dynamic operator simulates the behavior of a swarm of particles, where each particle is updated based on its current position, the best position found by the swarm, and the positions of its neighbors.
# The levy distribution is a probability distribution that is used to generate new positions for the particles.
# The probabilistic selector selects the operator with the highest probability of being chosen.
# The combination of these factors makes the SwarmMetaheuristic a suitable choice for optimizing the Rastrigin(2) function.