# Name: BohachevskyOptimization

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(3) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.012640636606838769,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0032700418189004618,
            'alpha': 0.005346405305791056,
            'beta': 1.559891064980173,
            'dt': 0.6152369399489788
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
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
# This metaheuristic combines a random search operator with central force dynamics to explore the solution space of the Bohachevsky function. The random search helps in escaping local minima, while the central force dynamic simulates the behavior of particles under attractive forces, potentially leading to faster convergence towards global optima. The Metropolis selector is used for the random search to allow uphill moves with a certain probability, ensuring exploration and escape from traps.