# Name: Adaptive Hybrid Metaheuristic
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

# Define the search operators and their parameters
heur = [
    (
        'random_sample',
        {},
        'greedy'
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
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
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
    )
]

# Initialize the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = False

# Run the metaheuristic with 30 iterations and record the fitness
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.run()
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines several search operators to explore the solution space more thoroughly. The 'random_sample' operator provides a broad initial exploration, while 'local_random_walk', 'spiral_dynamic', and 'swarm_dynamic' refine the search around promising regions. The use of different selectors ('greedy', 'probabilistic', 'metropolis') ensures a balanced exploration-exploitation strategy.

# If an error occurs, it is likely due to a misspelled operator name or missing dependencies. Double-check the operator names and ensure all required modules are installed.