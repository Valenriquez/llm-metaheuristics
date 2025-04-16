# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'central_force_dynamic',
        {
            'gravity': 0.0029354345813996452,
            'alpha': 0.01570115022704171,
            'beta': 2.5578777341571577,
            'dt': 0.8438416401676238
        },
        'metropolis'
    ),
    (
        'random_search',
        {
            'scale': 0.01414930829630552,
            'distribution': 'levy'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HybridMetaheuristic combines the Central Force Dynamic search operator with a Probabilistic Random Search operator. The Central Force Dynamic helps in efficiently searching the space by using a force model, while the Probabilistic Random Search provides additional exploration. This combination aims to balance exploitation and exploration, leading to better convergence and higher quality solutions for the Rastrigin function.