# Name: Custom Metaheuristic for Rastrigin Function
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
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
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The selected metaheuristic combines the `local_random_walk` and `swarm_dynamic` operators.
# The `local_random_walk` operator helps in exploring the local neighborhood of the current solution,
# which is useful for fine-tuning the solution near optima. It uses a uniform distribution for randomness.
#
# The `swarm_dynamic` operator, with an inertial version and Gaussian distribution, simulates the behavior
# of particles moving towards better solutions, mimicking social behavior like ants or birds. This helps in
# escaping local minima and exploring the global space efficiently.
#
# By combining both operators, we leverage their strengths: thorough exploration with `local_random_walk` 
# and fast convergence with `swarm_dynamic`, which should yield good performance for the Rastrigin function,
# especially given its complex landscape of multiple local minima.