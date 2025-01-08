# Name: Hybrid Metaheuristic Algorithm

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

# Define the search operators with their parameters and selectors
heur = [
    ('random_search', 
     {'scale': 1.0, 'distribution': 'uniform'}, 
     'greedy'),
    ('central_force_dynamic', 
     {'gravity': 0.001, 'alpha': 0.01, 'beta': 1.5, 'distribution': 'gaussian'}, 
     'probabilistic'),
    ('swarm_dynamic', 
     {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 
     'all'),
    ('spiral_dynamic', 
     {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 
     'metropolis')
]

# Initialize the metaheuristic with the problem and heuristics
met = mh.Metaheuristic(prob, heur, num_iterations=100)

# Run the metaheuristic for multiple iterations and collect fitness values
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    fitness.append(met.historical['fitness'])

# Convert fitness data to a numpy array and extract final fitness values
fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic algorithm combines the strengths of several search operators such as random search, central force dynamics, swarm intelligence, and spiral dynamics. The use of different operators allows the algorithm to explore various regions of the solution space efficiently. The probabilistic selector in the central force dynamic helps in balancing exploration and exploitation. The 'all' selector for swarm dynamics ensures that all particles are considered in each iteration, enhancing global search capabilities. Additionally, the metropolis selector is used in the spiral dynamic operator to allow uphill moves, which can help escape local optima and improve the overall solution quality.
# The algorithm runs multiple iterations (30 times) to ensure robustness and consistency of results. By collecting and analyzing the fitness values from each run, we can gain insights into the performance of the hybrid metaheuristic and its ability to solve complex optimization problems like the Rastrigin function.