# Name: Advanced Hybrid Metaheuristic for Global Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# - The metaheuristic combines two operators: Local Random Walk and Swarm Dynamic.
# - Local Random Walk helps in exploring the solution space locally, which is crucial for fine-grained search around promising regions.
# - Swarm Dynamic incorporates a social component that mimics the behavior of particles in a swarm to explore globally.
# - Both selectors are set to 'all' to ensure that both operators are used continuously throughout the optimization process.
# - This hybrid approach aims to balance exploration and exploitation, making it suitable for complex problems like Rastrigin.