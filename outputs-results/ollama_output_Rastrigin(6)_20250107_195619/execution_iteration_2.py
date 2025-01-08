# Name: Hybrid Metaheuristic Algorithm (HMA)
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

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
# The Hybrid Metaheuristic Algorithm (HMA) combines three different search operators: Random Search, Central Force Dynamic, and Swarm Dynamic. This combination allows the algorithm to explore a wide range of solutions while also benefiting from the local optimization capabilities of some operators.
# - **Random Search**: It is used as the initial exploration operator to ensure that the search space is thoroughly searched in the beginning.
# - **Central Force Dynamic**: This operator helps in guiding the search towards better regions by simulating forces between particles.
# - **Swarm Dynamic**: It is used for more fine-grained search in the promising areas identified by previous operators, leveraging the collective behavior of agents to find optimal solutions.
# The algorithm runs for 100 iterations with 30 replications to ensure robustness and reliability of the results.