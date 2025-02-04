# Name: Hybrid Metaheuristic for Rastrigin Function

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.6641174762764592,
            'self_conf': 2.9273095146212436,
            'swarm_conf': 2.5444744315877204,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',
        {
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
#met.verbose = True # please comment this line
#met.run() # please comment this line

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
# The Hybrid Metaheuristic combines the strengths of both swarm_dynamic and random_sample operators. The swarm_dynamic operator is designed to simulate the behavior of a flock or school of particles, which can efficiently explore the search space and converge to optimal solutions. However, it may get stuck in local optima. To avoid this, we incorporate the random_sample operator, which randomly samples points from the search space, helping to escape local optima and ensuring a more thorough exploration of the solution space. This hybrid approach aims to balance exploration and exploitation, leading to better performance on complex optimization problems like the Rastrigin function.