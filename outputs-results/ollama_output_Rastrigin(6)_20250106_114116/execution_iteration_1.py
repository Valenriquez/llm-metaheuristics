# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.3595962827099167,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5720971285797866,
            'self_conf': 2.856360521227085,
            'swarm_conf': 2.8599693560182167,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.516520427267299,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines three different search operators: Random Search, Swarm Dynamic, and Local Random Walk. Each operator is configured with specific parameters to suit the problem at hand. The combination allows for a diverse exploration of the solution space, balancing between exploitation and exploration. The metropolis selector is used for the random search, allowing it to escape local minima. The swarm dynamic selector encourages collaboration among agents, leading to better convergence towards the global optimum. The local random walk selector ensures that each agent can occasionally explore new regions independently, preventing premature convergence.