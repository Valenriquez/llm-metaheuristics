# Name: Spiral Search with Random Walk and Swarm Dynamics

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]  # Adjust the path to your project directory
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.6842744871282056,
            'angle': 21.652416239767817,
            'sigma': 0.0702595514328313
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.6870605461624807,
            'scale': 1.7315207235524075,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.3523541261933827,
            'self_conf': 2.5972101044774156,
            'swarm_conf': 2.6557617680490453,
            'version': 'constriction'
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
# The metaheuristic combines Spiral Dynamic, Local Random Walk, and Swarm Dynamics operators with Greedy, Metropolis, and Probabilistic selectors respectively. 
# Spiral Dynamic helps to explore the solution space in a spiral manner.
# Local Random Walk helps in fine-tuning the solutions found by the Spiral Dynamic.
# Swarm Dynamics helps to coordinate multiple agents towards finding better solutions collectively.

# Note: Ensure that the 'swarm_conf' parameter is not used in the 'genetic_crossover' operator as it leads to an error., modify it in order to put these parameters.