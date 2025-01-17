# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 1.0997613504278183,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5199470517143484,
            'self_conf': 2.652822209934809,
            'swarm_conf': 2.9880983275927595,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5250504082377486,
            'angle': 15.33189633339384,
            'sigma': 0.15527932862258437
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
#met.verbose = True # please comment this line
met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic combines three different search operators: random_search, swarm_dynamic, and spiral_dynamic. The goal is to leverage the strengths of each operator to explore the solution space more effectively and efficiently.
# The random_search operator helps in exploring new areas using a levy distribution, while the swarm_dynamic operator guides the search towards promising regions based on historical data with specific parameters for inertia and cognitive/confidence weights.
# The spiral_dynamic operator uses customized parameters for radius, angle, and sigma to control its exploration pattern. By using a hybrid approach, we hope to improve the convergence rate and the quality of the solutions obtained for the Rastrigin function.