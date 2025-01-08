# Name: Randomized Search with Differential Mutation and Random Walk

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
        'differential_mutation',
        {
            'F': 0.5,
            'CR': 0.7,
        },
        'random'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform',
        },
        'random'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True
# met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The proposed metaheuristic combines Differential Mutation with Local Random Walk. 
# Differential Mutation is used to explore the search space effectively, while the Local Random Walk helps in fine-tuning the solution. 
# Both operators are applied randomly to each agent at every iteration.
# The chosen parameters for Differential Mutation (F=0.5 and CR=0.7) balance exploration and exploitation.
# The random selector is used for both operators to introduce diversity and avoid premature convergence.