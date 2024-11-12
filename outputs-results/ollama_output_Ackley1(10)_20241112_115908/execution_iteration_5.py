# Name: Spiral_Ant
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

from benchmark_func import Ackley1
from metaheuristic import Metaheuristic

def spiral_ant():
    # This is the selected problem, the problem may vary depending on the case.
    fun = Ackley1(10)  
    prob = fun.get_formatted_problem()
    
    heur = [
        (  # Search operator 1
            'random_flight', 
            {
                'scale': 0.5,
                'distribution': 'levy',
                'beta': 2.5
            }, 
            'greedy'
        ),
        (  # Search operator 2
            'spiral_dynamic',  
            {
                'radius': 0.9,
                'angle': 20,  # Changed to 20 from 22.7 for better results
                'sigma': 0.1
            }, 
            'metropolis'
        ), 
        (  # Search operator 3
            'local_random_walk',  
            {
                'probability': 0.9,
                'scale': 1.0,
                'distribution': 'uniform'
            }, 
            'all'
        )
    ]
    
    met = Metaheuristic(prob, heur, num_iterations=100)
    met.verbose = True
    met.run()
    
    print('x_best = {}, f_best = {}'.format(*met.get_solution()))

if __name__ == "__main__":
    spiral_ant()