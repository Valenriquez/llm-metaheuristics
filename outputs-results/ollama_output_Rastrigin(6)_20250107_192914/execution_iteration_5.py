# Name: Adaptive Metaheuristic for Optimization

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the heuristics with varying operators and selectors
heur = [
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
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
        'greedy'
    ),
    (
        'random_sample',
        {},
        'metropolis'
    )
]

# Initialize the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)

# Run the metaheuristic with the same problem 30 times
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This adaptive metaheuristic combines multiple dynamic search operators, each with its own set of parameters and selector. The central force dynamic operator provides a smooth movement based on physics principles, while the spiral dynamic operator offers a more erratic path but maintains a coherent direction. The swarm dynamic approach mimics social behavior to explore the solution space efficiently. Finally, random sampling ensures that the search is not confined to any single region of the problem space.
# The 'all' selector allows each operator to choose freely among available moves, while others guide the selection process in specific ways (probabilistic, greedy). The metropolis selector helps escape local minima by allowing occasional worse solutions with a controlled probability. Running the metaheuristic multiple times helps assess its stability and effectiveness across different scenarios.