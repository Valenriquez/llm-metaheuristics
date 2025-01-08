# Name: Hybrid Metaheuristic with Adaptive Operator Selection

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

operators = [
    ("random_sample", {}, "greedy"),
    ("spiral_dynamic", {"radius": 0.585200258028537, "angle": 1.2379965825454224, "sigma": 0.5261224794554916}, "greedy"),
    ("swarm_dynamic", {"factor": 0.7, "self_conf": 2.54, "swarm_conf": 2.56, "version": "inertial", "distribution": "uniform"}, "metropolis"),
    ("local_random_walk", {"probability": 0.75, "scale": 1.0, "distribution": "gaussian"}, "probabilistic"),
    ("swarm_dynamic", {"factor": 1.0, "self_conf": 2.54, "swarm_conf": 2.56, "version": "constriction", "distribution": "uniform"}, "greedy")
]

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'spiral_dynamic',
        {'radius': 0.585200258028537, 'angle': 1.2379965825454224, 'sigma': 0.5261224794554916},
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True
# met.run()

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
# This metaheuristic combines several search operators with adaptive operator selection based on the performance history. It starts with a simple random sampling to explore the solution space, followed by spiral dynamic and swarm dynamics operators to refine the search process. The use of metropolis and probabilistic selectors helps in efficiently exploring the landscape and converging towards optimal solutions. The hybrid approach aims to balance exploration and exploitation, leading to better performance on various benchmark functions.