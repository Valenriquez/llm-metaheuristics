# Name: EvolSWF

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.2410407991798408,
            'self_conf': 2.0235176725424373,
            'swarm_conf': 2.4818549444247684,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.2718622139682812,
            'scale': 0.10476855066982281,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)  
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The EvolSWF metaheuristic combines the Swarm Dynamic (swarm_dynamic) and Local Random Walk (local_random_walk) operators. 
# The Swarm Dynamic operator is used to perform collective movement based on particle interactions with specific parameters, while the Local Random Walk operator introduces randomness for exploration using its own set of parameters.
# Both operators are selected with a 'metropolis' selector to ensure that they can explore new regions effectively.
# The algorithm runs for 1000 iterations and 30 repetitions, using 205 agents.