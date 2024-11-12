# Name: ackley_metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1({2: 20}) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Ackley function is a multi-modal optimization problem that can lead to premature convergence if not properly designed.
# In this metaheuristic, we use a combination of two search operators: local random walk and spiral dynamic. 
# Local random walk helps in finding the global minimum by randomly walking in the neighborhood of the current solution, 
# while spiral dynamic uses an iterative process to refine the solution and avoid getting stuck in local minima.
# The greedy and metropolis selection mechanisms are used to choose the next solution to explore based on their fitness values.