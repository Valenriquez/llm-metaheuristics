# Name: Random Walk Metaheuristic with Adjusted Parameters
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.5038073912664183,
            'scale': 0.10770532330701454,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
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
# The Random Walk Metaheuristic combines a random sampling operator with a local random walk operator.
# The random sampling operator provides initial solutions, while the local random walk operator explores the neighborhood of these solutions using adjusted parameters: probability=0.5038073912664183, scale=0.10770532330701454, and distribution='gaussian'.
# The Metropolis selector is used to decide whether to accept new solutions based on their fitness relative to the current solution.
# This combination helps in exploring the search space more effectively and finding better solutions with the specified parameters.