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
# The Swarm Metaheuristic with Inertial Version and Levy Distribution is chosen because it is a well-established metaheuristic that has been shown to perform well on the Rastrigin(2) optimization problem. The inertia version of the swarm metaheuristic allows the swarm to explore the search space more effectively, while the levy distribution provides a more efficient way to sample new candidate solutions.