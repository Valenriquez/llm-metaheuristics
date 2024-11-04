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
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The hybrid metaheuristic combines the swarm dynamics operator, which encourages exploration, and the spiral dynamics operator, which encourages exploitation.
# The swarm dynamics operator is selected using the metropolis selector, which favors the operator with the lowest energy state.
# The spiral dynamics operator is selected using the probabilistic selector, which randomly selects the operator with a probability proportional to its energy state.
# This combination of operators balances exploration and exploitation, resulting in a metaheuristic that is capable of finding good solutions to the Rastrigin function.