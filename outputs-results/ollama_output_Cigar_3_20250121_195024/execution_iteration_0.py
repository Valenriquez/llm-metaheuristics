# Name: CigarOptimizationMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(1) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5384700783312015,
            'self_conf': 2.477398542339028,
            'swarm_conf': 2.004308479932476,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6208829572204041,
            'angle': 13.60785680948365,
            'sigma': 0.19780007656525256
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
#met.verbose = True # please comment this line
met.run() # please comment this line

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
# The metaheuristic named CigarOptimizationMetaheuristic is designed to optimize the Cigar benchmark function using a combination of operators. We start with random sampling to explore the search space, then employ swarm dynamics for exploitation, and finally use spiral dynamics to guide the solution towards better regions. This multi-step approach aims to balance exploration and exploitation efficiently, leveraging the strengths of different search strategies in handling this specific optimization problem. The use of 30 iterations ensures robustness and reliability in finding optimal solutions for the Cigar benchmark function.