# Name: Adaptive Hybrid Metaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the search operators and their parameters
heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.01,
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
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 45.0,  # Adjusted angle for better performance
            'sigma': 0.1
        },
        'metropolis'
    )
]

# Initialize the metaheuristic with the problem and operators
met = mh.Metaheuristic(prob, heur, num_iterations=200)
#met.verbose = True # please comment this line
#met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=200, num_agents=150)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This adaptive hybrid metaheuristic combines three different search operators: random search, swarm dynamics, and spiral dynamics. 
# Each operator is configured with specific parameters to enhance exploration and exploitation during the optimization process.
# The selection of operators is based on their strengths in handling diverse problem landscapes and the ability to escape local optima effectively.
# The metropolis selector is used for the spiral dynamic operator to incorporate probabilistic acceptance criteria, which helps in exploring rugged search spaces more thoroughly.
# Running the metaheuristic 30 times with varying parameters ensures a robust analysis of the solution quality across multiple runs.