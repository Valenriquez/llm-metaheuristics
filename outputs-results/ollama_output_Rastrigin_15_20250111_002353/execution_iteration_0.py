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

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.6417721485470366,
            'self_conf': 2.5924199587667434,
            'swarm_conf': 2.992456719690023,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.838447545906818,
            'angle': 22.042102800755632,
            'sigma': 0.11817087132260369
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
#met.verbose = True # Uncomment this line to enable verbose mode.
#met.run() # Uncomment this line to run the metaheuristic.

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the advantages of both swarm dynamics and spiral dynamics. 
# Swarm dynamics, with its inertial weight and cognitive and social components, helps in exploring and exploiting the search space effectively.
# Spiral dynamics introduces a radial movement pattern that can help escape local minima more efficiently.
# The use of a greedy selector ensures that improvements are accepted quickly, while the probabilistic selector allows for exploration.
# The algorithm runs 30 times to gather statistics on the final fitness values, providing a robust assessment of its performance.