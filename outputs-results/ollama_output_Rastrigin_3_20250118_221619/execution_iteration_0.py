# Name: Hybrid Evolutionary Algorithm (HEA)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.30468562550773204,
            'distribution': 'levy'
        },
        'metropolis'
    ),
    (
        # Search operator 2: Genetic Crossover and Mutation
        'genetic_crossover',
        {
            'pairing': 'tournament_2_100',
            'crossover': 'linear_0.5_0.5',
            'mating_pool_factor': 0.12339318228264351
        },
        'probabilistic'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.21719727365676597,
            'mutation_rate': 0.5797134959378825,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Evolutionary Algorithm (HEA) combines a random search operator with genetic crossover and mutation operators. This hybrid approach aims to leverage the exploration capabilities of random search with the exploitation mechanisms of genetic algorithms. The metropolis selector for the random search ensures that better solutions are accepted, while the probabilistic selector for genetic operations allows for both exploration and exploitation in the solution space. The use of a levy distribution for the random search operator helps in efficiently exploring large regions of the solution space. The tournament selection method for genetic crossover promotes diversity by selecting parents based on their fitness levels. Linear crossover with a mixing factor of 0.5 ensures a balance between exploration and exploitation. Genetic mutation with a high mutation rate helps in escaping local optima. The metaheuristic is run multiple times to assess its performance on the Rastrigin function with a fixed number of agents and iterations.