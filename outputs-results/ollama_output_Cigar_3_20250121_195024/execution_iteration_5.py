# Name: Improved Simulated Annealing with Genetic Operators

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(3)
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.5629938782343802,
            'self_conf': 2.707874598548627,
            'swarm_conf': 2.9573972738020493,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'random_sample',
        {},
        'probabilistic'
    ),
    (
        'gravitational_search',
        {
            'gravity': 0.5422353857286947,
            'alpha': 0.035776608103870504
        },
        'greedy'
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
# The Improved Simulated Annealing with Genetic Operators combines the strengths of swarm dynamics, random sampling, and gravitational search. By integrating these operators, the algorithm aims to explore the solution space more effectively while maintaining a balance between exploration and exploitation. This approach is expected to improve convergence rates and final fitness values compared to using a single operator alone.