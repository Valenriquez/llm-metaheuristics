# Name: Modified Random Walk Metaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'probability': 0.5052813454371994,
            'scale': 0.14518870308205528,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Modified Random Walk Metaheuristic uses the local_random_walk operator with updated parameters to control its exploration behavior. The probability of performing a small random walk is set to 0.5052813454371994, ensuring a balanced exploration-explitation trade-off. The scale parameter is reduced to 0.14518870308205528 to allow for finer steps in the search space, potentially improving convergence towards the optimal solution. Additionally, the random_sample operator remains unchanged, continuing to introduce diversity into the population. This combination aims to enhance the overall performance of the metaheuristic on the Rastrigin function.