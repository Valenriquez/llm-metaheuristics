# Name: BohachevskyOptimizationMetaheuristic

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
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
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
# This metaheuristic combines a random search operator with a central force dynamic operator to explore the solution space of the Bohachevsky function. The random search helps in escaping local minima, while the central force dynamic provides guidance towards potentially better solutions. The Metropolis selector is used for the random search, allowing it to accept worse solutions occasionally, which aids in exploration. The probabilistic selector for the central force dynamic ensures that the dynamics are not deterministic but guided by probability, facilitating a balance between exploration and exploitation. Running multiple iterations and agents helps in refining the solution and mitigating the impact of initial conditions on the optimization outcome.