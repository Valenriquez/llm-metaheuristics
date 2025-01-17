# Name: Hybrid Metaheuristic for Rastrigin Function

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
    (
        'spiral_dynamic',
        {
            'radius': 0.6657732529290813,
            'angle': 22.704791784332212,
            'sigma': 0.043515665812604416
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5390496150592381,
            'self_conf': 2.75573639666415,
            'swarm_conf': 2.484772072133058,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'probabilistic'
    ),
    (
        'random_sample',
        {},
        'greedy'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the Spiral Dynamic, Swarm Dynamic, and Random Sample operators to explore different search spaces effectively. The Spiral Dynamic operator helps in escaping local minima by dynamically adjusting its radius and angle according to the given parameters. The Swarm Dynamic operator mimics the behavior of a social system, allowing particles to move towards better solutions based on their personal best and global best positions with specific confinement factors and distribution methods. The Random Sample operator ensures that the search is not trapped in any suboptimal regions. By combining these operators with the specified parameters, the hybrid metaheuristic can efficiently explore and exploit the solution space, leading to improved performance on the Rastrigin function.