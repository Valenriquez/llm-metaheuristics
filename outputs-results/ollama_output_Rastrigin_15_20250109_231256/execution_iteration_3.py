# Name: Randomized Swarm Dynamic Search with Constriction Version (RSDS-CV)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.579603331148707,
            'self_conf': 2.53519703198973,
            'swarm_conf': 2.731203898034554,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'random_sample',
        {},
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines the Swarm Dynamic Search (SDS) operator with a random sampling operator. The SDS operator is designed to simulate the behavior of social insects, while the random sampling operator helps in exploring the solution space more thoroughly. By combining these two operators, we aim to balance exploration and exploitation, leading to improved performance on benchmark functions like the Rastrigin function. The constriction version of the SDS ensures a more controlled search process, potentially enhancing convergence and stability.

# Note: If you encounter an error regarding 'position' key in the historical data, ensure that the `metaheuristic.py` file correctly handles the storage of positions and fitness values during the optimization process.