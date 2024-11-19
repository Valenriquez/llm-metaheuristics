# Name: Hybrid Metaheuristic for Rastrigin Function Optimization

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
        'random_sample',
        {
            'parameter1': None,
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "inertial",
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# - The `random_sample` operator is used to explore the search space randomly, providing an initial exploration phase.
# - The `local_random_walk` operator helps refine solutions by performing small random steps around the current solution.
# - The `swarm_dynamic` operator mimics the behavior of a swarm (e.g., birds or fish) where each individual (solution) influences the group's movement based on personal best and social best positions. This operator is particularly useful in higher dimensions due to its probabilistic nature, which can help escape local optima.
# - The 'probabilistic' selector allows each operator to execute with a certain probability, enhancing exploration by switching between different strategies.

# This hybrid approach aims to combine global search capabilities (random sampling) with local refinement and swarm-based behavior, making it suitable for optimizing the Rastrigin function in higher dimensions.