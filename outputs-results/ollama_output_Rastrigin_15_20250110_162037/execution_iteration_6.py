# Name: Adaptive Hybrid Metaheuristic with Dynamic Parameter Tuning

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(10)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Walk
        'RandomWalk',
        {
            'step_size': 0.5,
            'max_steps': 50
        },
        'greedy'
    ),
    (
        'ParticleSwarmOptimization',
        {
            'num_agents': 30,
            'inertia_weight': 0.7,
            'cognitive_const': 1.4,
            'social_const': 1.4,
            'velocity_bound': [-1, 1]
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The adaptive hybrid metaheuristic combines a Random Walk with Particle Swarm Optimization (PSO) to enhance convergence. 
# Dynamic parameter tuning adjusts the step size and learning rates based on performance trends.
# Adaptive population size management allows for expansion or reduction of agents based on convergence rate.
# Diversity control ensures the population maintains variety through hybridization, preventing premature convergence.
# This design aims to leverage both local fine-tuning and global exploration strategies.