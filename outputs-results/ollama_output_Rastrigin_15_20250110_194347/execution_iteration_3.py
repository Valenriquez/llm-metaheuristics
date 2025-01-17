# Name: Hybrid Metaheuristic with Spiral and Swarm Dynamics

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.745259582972196,
            'angle': 20.360127679398126,
            'sigma': 0.010268674430911187
        },
        'all'
    ),
    (
        # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.8964692117678443,
            'self_conf': 2.1053772175074377,
            'swarm_conf': 2.817536458632691,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'all'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=20)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines Spiral Dynamic and Swarm Dynamic operators. The Spiral Dynamic operator is used to explore the search space by following a spiral path, while the Swarm Dynamic operator is used to exploit local optima by mimicking the behavior of particles in a swarm. Both operators are applied with the 'all' selector, which applies them to all agents at every iteration. This combination aims to balance exploration and exploitation effectively for the Rastrigin function, which has many local minima but a global minimum at the origin. The use of both operators helps in efficiently navigating through the complex landscape of the Rastrigin function.