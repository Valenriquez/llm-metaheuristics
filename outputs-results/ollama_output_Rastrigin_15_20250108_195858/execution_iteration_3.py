# Name: Hybrid Metaheuristic with Variable Selection

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

# Define the operators and their parameters
operators = [
    ('random_sample', {}, 'greedy'),
    ('local_random_walk', {'scale': 0.5, 'probability': 0.75, 'distribution': 'uniform'}, 'greedy'),
    ('swarm_dynamic', {'factor': 0.8, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'gaussian'}, 'metropolis'),
    ('spiral_dynamic', {'radius': 0.9, 'angle': 18, 'sigma': 0.05}, 'probabilistic')
]

heur = [
    (op[0], op[1], op[2]) for op in operators
]

# Initialize the Metaheuristic with a list of operators and their parameters
met = mh.Metaheuristic(prob, heur, num_iterations=1000)

# Run the metaheuristic 30 times and collect fitness values
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic incorporates multiple operators with varying parameters and selection strategies (greedy, metropolis, probabilistic) to explore the solution space more effectively. The `random_sample` operator provides a broad search, while `local_random_walk` refines solutions around the current best. `swarm_dynamic` uses a particle swarm approach for convergence, and `spiral_dynamic` helps in escaping local minima by spiraling outward. This combination aims to balance exploration and exploitation, enhancing the overall performance on complex benchmark functions like Rastrigin.