# Name: Adaptive Hybrid Metaheuristic (AHM)

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem
prob = fun.get_formatted_problem()

# Define a schedule for operator switching
operator_schedule = [
    "random_search",
    "central_force_dynamic",
    "differential_mutation",
    "firefly_dynamic",
    "gravitational_search",
    "random_flight",
    "local_random_walk",
    "spiral_dynamic",
    "swarm_dynamic"
]

heur = []
for operator in operator_schedule:
    if operator == "random_search":
        heur.append(
            (operator, {
                'scale': 1.0,
                'distribution': 'uniform'
            }, 'greedy')
        )
    elif operator == "central_force_dynamic":
        heur.append(
            (operator, {
                'gravity': 0.001,
                'alpha': 0.01,
                'beta': 1.5,
                'dt': 1.0
            }, 'probabilistic')
        )
    elif operator == "differential_mutation":
        heur.append(
            (operator, {
                'parameter1': 'mutation_rate',
                'value1': 0.5
            }, 'greedy')
        )
    elif operator == "firefly_dynamic":
        heur.append(
            (operator, {
                'sigma': 0.2,
                'beta_min': 0.1,
                'beta_max': 1.5
            }, 'metropolis')
        )
    elif operator == "gravitational_search":
        heur.append(
            (operator, {
                'G': 6.67430e-11,
                'C1': 2.0,
                'C2': 2.0
            }, 'greedy')
        )
    elif operator == "random_flight":
        heur.append(
            (operator, {
                'speed': 0.5,
                'angle': 45
            }, 'probabilistic')
        )
    elif operator == "local_random_walk":
        heur.append(
            (operator, {
                'step_size': 0.1,
                'num_steps': 10
            }, 'greedy')
        )
    elif operator == "spiral_dynamic":
        heur.append(
            (operator, {
                'radius': 0.9,
                'angle': 22.5,
                'sigma': 0.1
            }, 'probabilistic')
        )
    elif operator == "swarm_dynamic":
        heur.append(
            (operator, {
                'factor': 0.7,
                'self_conf': 2.54,
                'swarm_conf': 2.56,
                'version': 'inertial',
                'distribution': 'uniform'
            }, 'greedy')
        )

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Adaptive Hybrid Metaheuristic (AHM) combines multiple search operators with varying parameters to explore the solution space more effectively. By dynamically switching between these operators, AHM aims to balance exploration and exploitation, leading to improved convergence rates and better solutions for complex optimization problems like Rastrigin's function.