## Name: Swarm Metaheuristic with Gaussian Noise

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
# The Swarm Metaheuristic with Gaussian Noise is chosen as it combines the benefits of the swarm optimization with the probabilistic nature of Gaussian noise. This approach encourages diversity in the search space while maintaining a focus on promising areas. The specific parameters are chosen based on the recommendations in parameters_to_take.txt. The factor value of 0.7 promotes exploration, while the self-confidence and swarm-confidence values ensure stability. The inertial version of the swarm algorithm provides a smooth and efficient search trajectory. The Gaussian distribution ensures that the noise is distributed in a way that complements the swarm dynamics, further enhancing the search efficiency.