# Name: Hybrid Evolutionary Metaheuristic (HEM)
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.032685895892821774,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.003086271363574029,
            'alpha': 0.7044068163305478,
            'beta': 2.4269991684998464
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.786209451833629,
            'self_conf': 2.632657877572002,
            'swarm_conf': 2.998580020361624,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8187779337204333,
            'angle': 24.632145542494907,
            'sigma': 0.13463559923135474
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8903041548898911,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

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
# The Hybrid Evolutionary Metaheuristic (HEM) combines several different search operators, each with unique properties that allow it to explore the solution space in distinct ways. By using a combination of these operators, HEM aims to leverage their strengths to find high-quality solutions more efficiently.
# 
# The `random_search` operator helps in exploring new regions of the solution space by randomly selecting solutions, which is useful for avoiding premature convergence.
# 
# The `central_force_dynamic` operator mimics the gravitational forces between particles in a system, guiding agents towards better solutions based on their historical interactions and the force exerted by other agents.
# 
# The `swarm_dynamic` operator models the behavior of a swarm to explore the solution space collectively. The 'constriction' version ensures that the swarm converges efficiently while maintaining diversity, allowing for a thorough exploration of the search space.
# 
# The `spiral_dynamic` operator encourages agents to move in a spiral pattern around promising regions of the solution space, enhancing the ability to find fine-grained solutions.
# 
# The `local_random_walk` operator allows agents to perform random walks in their local neighborhood, helping them escape from local optima and explore diverse parts of the search space. This is particularly useful for global optimization problems.
# 
# By combining these operators with the specified parameters, HEM aims to provide a robust and effective approach for solving complex optimization problems.