# Name: Hybrid Metaheuristic with Random Sample and Swarm Dynamic

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'greedy'),
    (
        'swarm_dynamic',
        {
            'factor': 0.5619137418882504,
            'self_conf': 2.732585582071534,
            'swarm_conf': 2.273283928594351,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
# met.verbose = True # Uncomment this line if you want to see verbose output
# met.run() # Uncomment this line to run the metaheuristic

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
# The hybrid metaheuristic combines a random sample operator with a swarm dynamic operator. 
# The random sample operator helps to explore the solution space more thoroughly, while the swarm dynamic operator helps to exploit promising regions by simulating the behavior of particles in a swarm.
# By combining these two operators, the hybrid metaheuristic aims to balance exploration and exploitation, potentially leading to better solutions for the Rastrigin function.