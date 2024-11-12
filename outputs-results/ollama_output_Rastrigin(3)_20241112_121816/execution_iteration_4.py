# Rastrigin Metaheuristic
# =====================
#
# This metaheuristic uses the Rastrigin function with 3 dimensions. It combines two search operators:
#   - `local_random_walk`: a local search operator that moves randomly in the current direction.
#   - `spiral_dynamic`: a global search operator that moves along a spiral path.
#
# The probability of moving in each iteration is set to 0.5 for the `local_random_walk` operator, and
# its parameters are adjusted so that it converges more quickly than the `spiral_dynamic` operator with the default settings.

import sys
from pathlib import Path

def rastrigin_metaheuristic():
    # Set project directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Import required modules
    sys.path.insert(0, str(project_dir))
    from benchmark_func import Rastrigin
    from metaheuristic import Metaheuristic
    
    # Define the function to optimize (Rastrigin)
    fun = Rastrigin(3)
    
    # Get the formatted problem
    prob = fun.get_formatted_problem()
    
    # Define search operators and their parameters
    heur = [
        (
            'local_random_walk',
            {
                'probability': 0.5,
                'scale': 1.0,
                'distribution': 'levy',
                'beta': 2,
            },
            'greedy'
        ),
        (
            'spiral_dynamic',
            {
                'radius': 0.99,   # Decreased radius to converge more efficiently
                'angle': 22.5,
                'sigma': 0.05,
            },
            'random_sample'
        )
    ]
    
    # Create the metaheuristic and run it
    met = Metaheuristic(prob, heur, num_iterations=100)
    met.verbose = True
    met.run()
    
    # Print the best solution found
    print('x_best = {}, f_best = {}'.format(met.get_solution()))

rastrigin_metaheuristic()