# Name: Hybrid Metaheuristic with Swarm and Spiral Dynamics

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
        {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'gaussian'},
        'probabilistic'
    ),
    ('spiral_dynamic',
        {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1},
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the swarm and spiral dynamics operators.
# The swarm operator simulates the behavior of a swarm of particles, where the direction of each particle is determined by the positions of its neighbors.
# The spiral operator explores the search space in a spiral-like manner, where the radius and angle of the spiral can be adjusted to control the exploration-exploitation balance.
# The probabilistic selector is used with the swarm operator to guide the direction of the particles, while the metropolis selector is used with the spiral operator to determine the probability of accepting new solutions.