# Name: BohachevskySearch

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
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.024773482522906315,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.9713687210890819,
            'self_conf': 2.593020622462789,
            'swarm_conf': 2.565443788412318,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8617493036276037,
            'angle': 21.576645157176337,
            'sigma': 0.06024435153889191
        },
        'all'
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
# The BohachevskySearch metaheuristic combines the strengths of different search operators to explore the solution space effectively. 
# The random_search operator helps in escaping local optima, while the swarm_dynamic and spiral_dynamic operators guide the population towards better solutions.
# This hybrid approach is expected to perform well on the Bohachevsky function, which has a single global minimum surrounded by multiple local minima.