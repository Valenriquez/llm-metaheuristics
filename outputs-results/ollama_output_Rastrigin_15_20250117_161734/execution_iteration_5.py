# Name: Enhanced Hybrid Metaheuristic (EHM)

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.8110128978321639,
            'self_conf': 2.2867631704006275,
            'swarm_conf': 2.9545089093107033,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8217443540291389,
            'angle': 21.02930445612904,
            'sigma': 0.10214115387684122
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Enhanced Hybrid Metaheuristic (EHM) combines the strengths of two different search operators: Swarm Dynamic and Spiral Dynamic. 
# The Swarm Dynamic operator is effective in exploring the solution space globally, while the Spiral Dynamic operator focuses on refining solutions locally.
# By integrating these operators with the specified parameters, EHM aims to balance exploration and exploitation effectively, potentially leading to better convergence towards optimal solutions.