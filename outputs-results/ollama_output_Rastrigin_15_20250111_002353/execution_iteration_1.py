# Name: Custom Metaheuristic with Random Search and Swarm Dynamic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.1311674029668554,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5020998916211454,
            'self_conf': 1.8090699777052455,
            'swarm_conf': 2.4324194755998785,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
# met.verbose = True
# met.run()

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
# This metaheuristic combines Random Search with Swarm Dynamic to explore the solution space effectively. 
# The random search operator helps in exploring new areas of the solution space, while the swarm dynamic operator leverages the collective intelligence of agents.
# The metropolis selector is used for accepting or rejecting candidate solutions based on their fitness values and a cooling schedule.