# Name: Hybrid Swarm Optimization with Random Walks (HSORW)

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.6821750309543491,
            'self_conf': 2.6615641248305932,
            'swarm_conf': 1.5860358017478784,
            'version': "constriction",
            'distribution': "uniform"
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.7622265827529788,
            'scale': 0.13520395004643035,
            'distribution': "uniform"
        },
        'greedy'
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
# HSORW combines the global search capabilities of swarm optimization with the fine-tuning ability of local random walks. The swarm_dynamic operator ensures that agents can explore the solution space effectively, while local_random_walk helps refine the solutions found by the swarm, potentially leading to better convergence. This hybrid approach leverages the strengths of both methods to improve overall performance on the Rastrigin function.