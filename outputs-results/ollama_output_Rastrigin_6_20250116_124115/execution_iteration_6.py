# Name: Hybrid Swarm Optimization with Adaptive Operator Selection

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Flight
        'random_flight',
        {
            'scale': 0.14044771279078613,
            'distribution': 'levy',
            'beta': 1.6282111624029503
        },
        'greedy'
    ),
    (
        # Search operator 2: Swarm Dynamic with Inertial Version
        'swarm_dynamic',
        {
            'factor': 0.5288233576985366,
            'self_conf': 2.637242399489272,
            'swarm_conf': 2.538614986224833,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        # Search operator 3: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.10660182673765349,
            'angle': 5.313448289570067,
            'sigma': 0.4933558493865862
        },
        'greedy'
    ),
    (
        # Search operator 4: Local Random Walk with Gaussian Distribution
        'local_random_walk',
        {
            'probability': 0.40167436245427723,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid swarm optimization metaheuristic combines the strengths of different search operators. The random flight operator helps in exploring new regions with a levy distribution, while the swarm dynamic operator leverages social interactions to converge towards the optimal solution using an inertial version. The spiral dynamic operator provides a balance between exploration and exploitation, and the local random walk ensures fine-tuning around the best solutions with a Gaussian distribution. This combination is expected to perform well on the Rastrigin function due to its global and rugged landscape.