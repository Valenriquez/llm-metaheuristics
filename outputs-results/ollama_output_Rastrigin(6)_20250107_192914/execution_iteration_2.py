# Name: Hybrid Metaheuristic Algorithm

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the heuristics with their parameters and selectors
heur = [
    (
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
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
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    )
]

# Initialize the metaheuristic with the problem and heuristics
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
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic algorithm combines the strengths of three different search operators: Random Search, Swarm Dynamic, and Spiral Dynamic. The Random Search operator helps explore the solution space randomly with a small scale and uniform distribution to avoid getting stuck in local optima. The Swarm Dynamic operator utilizes particle swarm optimization principles with an inertial version for more robust exploration. The Spiral Dynamic operator leverages spiral movement to efficiently search around promising regions. By combining these operators, the algorithm can leverage their individual strengths to find better solutions for the Rastrigin function problem.