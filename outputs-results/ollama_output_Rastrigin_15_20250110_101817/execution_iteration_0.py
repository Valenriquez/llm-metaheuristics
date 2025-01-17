# Name: Hybrid Metaheuristic for Optimization

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.2347731572675416,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.15272749210033187,
            'alpha': 0.5129532214008398,
            'beta': 1.7672887323585988
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8630179474615391,
            'self_conf': 2.7461653191276594,
            'swarm_conf': 2.685637406633233,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
#met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000)
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The hybrid metaheuristic combines multiple search operators to enhance the exploration and exploitation capabilities of the optimization process. The random_search operator helps in exploring new areas with a larger scale and uniform distribution, central_force_dynamic promotes convergence towards the optimal solution using adjusted gravity, alpha, and beta values, swarm_dynamic encourages social learning among agents with refined parameters for factor, self_conf, and swarm_conf, and random_sample ensures a broad exploration. This combination aims to balance exploration and exploitation effectively, leading to better and more robust solutions for the given optimization problem.