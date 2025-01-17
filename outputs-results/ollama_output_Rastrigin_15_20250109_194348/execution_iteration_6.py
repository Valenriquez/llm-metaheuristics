# Name: HarmonySearchWithRandomFlight
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_flight',
        {
            'scale': 0.8968084681545709,
            'distribution': 'uniform',
            'beta': 2.1196653397047034
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6766185557924459,
            'self_conf': 2.6249540530983704,
            'swarm_conf': 2.0172725902428326,
            'version': 'constriction',
            'distribution': 'gaussian'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines Random Flight and Swarm Dynamic operators. 
# Random Flight helps to explore the search space by making large, random movements.
# Swarm Dynamic operator is used to guide the population towards promising regions,
# which helps in refining the solution. The combination of both operators allows for
# a balance between exploration and exploitation, enhancing the overall performance
# of the metaheuristic on the Rastrigin function. The specified parameters were chosen
# to optimize the search process, ensuring efficient exploration and effective
# exploitation of the search space.