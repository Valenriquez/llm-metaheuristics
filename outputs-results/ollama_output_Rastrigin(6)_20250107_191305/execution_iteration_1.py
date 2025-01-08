# Name: Hybrid Metaheuristic Approach

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.5,
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.01,
            'alpha': 0.02,
            'beta': 1.3,
            'version': 'constriction'
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.85,
            'self_conf': 2.4,
            'swarm_conf': 2.6,
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
#met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=150, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic approach combines three different operators to potentially improve the search efficiency of the Rastrigin function. The `random_search` operator provides initial exploration, while the `central_force_dynamic` and `swarm_dynamic` operators help in refining the solution and avoiding premature convergence. The use of 'all' selector for random search ensures that all agents can utilize this method during every iteration, leading to diversified exploration. Similarly, the probabilistic and greedy selectors for central force dynamic and swarm dynamic respectively ensure that these operators are used effectively without locking the system into a single behavior. This combination is expected to provide a robust solution for the Rastrigin function.