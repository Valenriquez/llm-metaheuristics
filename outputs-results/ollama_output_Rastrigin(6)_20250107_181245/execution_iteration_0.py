# Name: Hybrid Metaheuristic Algorithm
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)
prob = fun.get_formatted_problem()

# Define the heuristic operators and selectors
heur = [
    ('random_search', 
     {'scale': 1.0, 'distribution': 'uniform'}, 
     'greedy'),
    
    ('central_force_dynamic', 
     {'gravity': 0.001, 'alpha': 0.01, 'beta': 1.5, 'dt': 1.0}, 
     'metropolis'),
    
    ('swarm_dynamic', 
     {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 
     'probabilistic'),
    
    ('spiral_dynamic', 
     {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 
     'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)

# Initialize the fitness register
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
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic Algorithm combines the strengths of several search operators with different selectors to enhance exploration and exploitation capabilities. Each operator is chosen based on its suitability for the problem at hand, aiming to balance exploration and convergence. The use of multiple operators and selectors helps in avoiding premature convergence and encourages a more thorough search of the solution space.