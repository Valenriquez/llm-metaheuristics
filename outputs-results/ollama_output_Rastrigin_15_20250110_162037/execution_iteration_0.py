# Name: Hybrid Metaheuristic Algorithm

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.07227683036632991,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.007235458348182218,
            'alpha': 0.07132130015699502,
            'beta': 4.300256807355229,
            'dt': 1.0010712373535615
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7300173446047333,
            'self_conf': 2.9914127374079515,
            'swarm_conf': 2.8071137931678485,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8017988926444446,
            'angle': 21.328865487110964,
            'sigma': 0.05579892819016799
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
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
# This hybrid metaheuristic algorithm combines the strengths of different optimization techniques to improve the search process. The random_search operator helps in exploring the solution space, while the central_force_dynamic operator simulates a physical system that moves towards the best solutions with adjusted parameters for enhanced performance. The swarm_dynamic operator models social behavior effectively with specified self-confidence and swarm confidence levels, encouraging agents to work together and learn from each other. The spiral_dynamic operator enhances exploration using a modified radius, angle, and sigma value, contributing to more efficient search across the problem space. Together, these operators provide a comprehensive search mechanism that can effectively solve complex optimization problems like Rastrigin's function with improved convergence rates and solution quality.