# Name: Swarm-Based Metaheuristic with Inertial Version and Levy Distribution

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
    (  # Swarm operator with inertial version and levy distribution
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
# This metaheuristic combines the swarm operator with the inertial version for efficient exploration and the levy distribution for diversification.
# The probabilistic selector ensures that the swarm operator is applied with a probability based on the levy distribution, leading to a balance between exploration and exploitation.
# The parameters are chosen based on the recommendations in the parameters_to_take.txt file for the swarm operator with inertial version and levy distribution.