# Name: Hybrid Metaheuristic Algorithm

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem.
prob = fun.get_formatted_problem()

# Define a combination of operators and selectors based on their strengths
heur = [
    (
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.005,
            'alpha': 0.02,
            'beta': 1.3
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.9,
            'self_conf': 2.7,
            'swarm_conf': 2.58,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 30,
            'sigma': 0.2
        },
        'probabilistic'
    )
]

# Initialize and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = False
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic Algorithm combines the strengths of various operators to improve exploration and exploitation. The 'random_search' operator helps in exploring new regions, while 'central_force_dynamic', 'swarm_dynamic', and 'spiral_dynamic' enhance the ability to exploit promising areas efficiently. Each operator is paired with a selector that suits its behavior, such as 'all' for randomness and 'probabilistic' for guided exploration. This combination allows for a balanced approach to optimization, balancing between exploring new solutions and refining existing ones.