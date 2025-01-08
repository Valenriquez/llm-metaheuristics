# Name: Hybrid Swarm-Local Search Metaheuristic (HSLSM)
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

heur = [
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
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Swarm-Local Search Metaheuristic (HSLSM) combines the strengths of swarm-based optimization with local search techniques. The `swarm_dynamic` operator initializes a population of agents to explore the solution space globally, while the `local_random_walk` operator refines the solutions locally by making small random changes. This hybrid approach leverages global exploration and local exploitation, potentially leading to better convergence and higher quality solutions for complex benchmark functions like Rastrigin.

# Addressing the error:
# The error encountered suggests that the historical data does not contain a key named 'position'. This could be due to several reasons, such as an incorrect implementation of the `get_solution` method in the `metaheuristic.py` file or issues with the way historical data is recorded and retrieved. To resolve this, one should check the implementation of the `Metaheuristic` class and ensure that all necessary historical data keys are properly initialized and updated throughout the execution of the metaheuristic algorithm.