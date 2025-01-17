# Name: Custom Metaheuristic for Rastrigin Function
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',  # Search operator 1
        {},
        'greedy'
    ),
    (
        'local_random_walk',  # Search operator 2
        {
            'probability': 0.5157124503025801,
            'scale': 0.9001047975943401,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',  # Search operator 3
        {
            'factor': 0.5921190419496202,
            'self_conf': 2.9604179581763437,
            'swarm_conf': 2.3040657939317346,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines three different search operators to explore the Rastrigin function. The random_sample operator provides initial randomness, while the local_random_walk and swarm_dynamic operators refine and balance the exploration and exploitation phases. The use of different selectors (greedy, metropolis, probabilistic) helps in managing the trade-off between diversity and convergence, leading to a robust search strategy for the Rastrigin function.