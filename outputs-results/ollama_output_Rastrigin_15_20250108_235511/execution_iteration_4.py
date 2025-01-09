# Name: Hybrid Metaheuristic with Diversified Operators

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem.
prob = fun.get_formatted_problem()

# Define a hybrid set of operators with diversified strategies
heur = [
    (
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.002,
            'alpha': 0.95
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 15,
            'sigma': 0.1
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
# met.verbose = True  # Uncomment this line if you want to see the output during the run.
# met.run()  # Uncomment this line if you want to run the metaheuristic.

# Initialize the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=200, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

# Convert the fitness list to a numpy array for easier manipulation
fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines several diverse operators, each with its own set of parameters and selection strategy.
# The 'random_search' operator helps to explore the solution space randomly but efficiently.
# The 'central_force_dynamic' operator mimics the behavior of charged particles in a physical system, which can lead to rapid convergence.
# The 'swarm_dynamic' operator uses the concept of swarms, such as bees or birds, for exploration and exploitation.
# The 'spiral_dynamic' operator follows a spiral trajectory to explore the solution space systematically.
# By using a combination of these operators, we aim to balance exploration and exploitation, leading to better overall performance on the problem.