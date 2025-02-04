# Name: Hybrid Metaheuristic for Cigar Function

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(1)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 1.241237844679017,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0073995293553142305,
            'alpha': 0.03275684950906025,
            'beta': 1.8389144533313646,
            'dt': 1.5905487750103424
        },
        'metropolis'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 2,
            'factor': 0.9171114754099607
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
# met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines three different operators: Random Search, Central Force Dynamic, and Differential Mutation. 
# Random Search helps in exploring the solution space thoroughly with a levy distribution, Central Force Dynamic is good for guiding the search towards the optimal region using specified parameters, 
# and Differential Mutation enhances the exploration by leveraging the differences between randomly selected individuals. The combination of these
# operators allows for a more robust exploration and exploitation of the problem space, leading to better optimization results.