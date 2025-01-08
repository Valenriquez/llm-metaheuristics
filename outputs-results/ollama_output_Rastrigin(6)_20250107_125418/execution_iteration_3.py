# Name: Hybrid Metaheuristic (HMH)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.038082469852601546,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.00314943410997236,
            'alpha': 0.9836992991618201,
            'beta': 2.1056964872151736
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6195821214192347,
            'self_conf': 2.248914005734117,
            'swarm_conf': 2.419838211226165,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic (HMH) combines three different search operators to explore the solution space more effectively. 
# 'random_search' is used for initial exploration, 'central_force_dynamic' helps in focusing on promising regions, and 'swarm_dynamic' mimics the behavior of a swarm to exploit local optima.
# The 'metropolis', 'greedy', and 'probabilistic' selectors ensure that the algorithm balances exploration and exploitation, making it robust for global optimization problems like Rastrigin's function.