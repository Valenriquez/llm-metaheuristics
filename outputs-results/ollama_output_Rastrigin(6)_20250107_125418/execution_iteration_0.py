# Name: Hybrid Metaheuristic (HMM)
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the search operators and their respective parameters and selectors
heur = [
    (
        'random_search',
        {
            'scale': 0.1762394455863434,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0028534381782677905,
            'alpha': 0.09905665781767779,
            'beta': 3.83228076872624
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5843890307412092,
            'self_conf': 2.17539308915773,
            'swarm_conf': 2.1766408917758766,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8233890789859447,
            'angle': 21.65118895444667,
            'sigma': 0.21990570640578527
        },
        'probabilistic'
    )
]

# Initialize and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Run the metaheuristic with the same problem 30 times
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.run()
    fitness.append(met.historical['fitness'])

# Convert fitness data to numpy array and extract final fitness values
fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic (HMM) combines multiple search operators with diverse characteristics to explore the solution space more effectively. By leveraging different operators, the HMM aims to balance exploration and exploitation, leading to a more robust optimization process. Each operator is assigned a specific selector based on its behavior: 'greedy' for quick convergence, 'all' for comprehensive exploration, 'metropolis' for probabilistic acceptance of worse solutions, and 'probabilistic' for stochastic search. The combination of these operators helps the HMM adapt to the problem's characteristics, potentially leading to higher quality solutions.