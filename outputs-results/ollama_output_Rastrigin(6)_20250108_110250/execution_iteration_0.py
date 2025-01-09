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

heur = [
    (
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'greedy'
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
# This hybrid metaheuristic algorithm combines three different search operators: Random Search, Central Force Dynamics, and Swarm Dynamic. The Random Search operator helps in exploring the solution space, the Central Force Dynamics operator guides particles towards potential optima by simulating gravitational effects, and the Swarm Dynamic operator encourages cooperation among agents to find better solutions.
# The use of 'all' selector for Random Search ensures that each agent explores the entire search space thoroughly, while the 'greedy' selector for Central Force Dynamics allows each agent to make locally optimal decisions, which can accelerate convergence. The 'probabilistic' selector for Swarm Dynamic helps in maintaining a diverse set of solutions and avoiding premature convergence.
# The metaheuristic is run 30 times with different initial conditions to evaluate its robustness and average performance on the Rastrigin function.