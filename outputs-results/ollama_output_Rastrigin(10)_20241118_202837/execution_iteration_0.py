# Name: Adaptive Metaheuristic for Rastrigin Function (AMH-Rastrigin)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
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
# AMH-Rastrigin is an adaptive metaheuristic designed to effectively explore the search space of the Rastrigin function.
# It combines the local search capabilities of the 'local_random_walk' operator with the global exploration power of the 'swarm_dynamic' operator.
# The 'local_random_walk' operator helps in fine-tuning the solution by making small, random steps based on a given probability, ensuring that the algorithm can escape from local minima.
# The 'swarm_dynamic' operator, using its parameters, guides the search towards more promising regions of the search space, leveraging global information for better convergence.
# The use of the 'greedy' selector in the first operator ensures that the algorithm makes progress towards improving solutions quickly. The 'all' selector for the second operator allows for a thorough exploration across the entire problem domain, enhancing the chances of finding the global optimum.