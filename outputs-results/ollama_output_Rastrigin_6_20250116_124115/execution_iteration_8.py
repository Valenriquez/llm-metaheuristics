# Name: Hybrid Swarm Optimization (HSO)

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
    (  # Search operator 1: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.6191684107283848,
            'self_conf': 2.197380536262161,
            'swarm_conf': 2.0167912614026835,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        # Search operator 2: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.5440070285932872,
            'scale': 0.9899196294315602,
            'distribution': 'gaussian'
        },
        'probabilistic'
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
# Hybrid Swarm Optimization (HSO) combines the global search capabilities of swarm dynamics with the local exploration power of a local random walk. This hybrid approach aims to balance exploration and exploitation, leading to potentially better convergence properties on complex optimization landscapes like Rastrigin's function.
# The swarm dynamic operator helps in efficiently exploring the solution space globally, while the local random walk operator refines solutions by making small, probabilistic steps. Together, these operators contribute to a robust search strategy that can handle both smooth and rugged fitness landscapes effectively.