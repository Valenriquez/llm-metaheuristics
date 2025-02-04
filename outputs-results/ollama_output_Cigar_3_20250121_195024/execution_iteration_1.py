# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(1)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.1590396214820446,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.004826682438526454,
            'alpha': 0.0491577274291412,
            'beta': 1.815752026049918,
            'dt': 1.2948069196654322
        },
        'greedy'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand',
            'num_rands': 2,
            'factor': 0.814442858822594
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()

    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of three different search operators: Random Search, Central Force Dynamic, and Differential Mutation. 
# The Random Search helps in exploring the solution space thoroughly with a scale of 0.159 and a uniform distribution, while Central Force Dynamic is used to guide the agents towards more promising areas with specific gravity, alpha, beta, and dt values.
# Differential Mutation introduces diversity with 'rand' expression, two random vectors, and a factor of 0.814, helping escape local optima. The combination allows for a robust exploration and exploitation of the problem space,
# leading to better convergence and higher quality solutions compared to using any single operator alone.