# Name: Hybrid Swarm-Genetic Metaheuristic (HSGM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.703915969274305,
            'self_conf': 2.534063761525474,
            'swarm_conf': 2.6026479644246505,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'genetic_crossover',
        {
            'pairing': 'random',
            'crossover': 'linear_0.5_0.5',
            'mating_pool_factor': 0.3453262082487612
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Swarm-Genetic Metaheuristic (HSGM) combines the strengths of both swarm intelligence and genetic algorithms. 
# The swarm_dynamic operator is used to guide the search through the solution space, utilizing the collective behavior of agents 
# to explore and exploit promising regions. The genetic_crossover operator further refines the solutions by combining the genetic information
# from multiple agents, promoting diversity and encouraging exploration while also improving the quality of solutions.
# This hybrid approach allows HSGM to efficiently balance exploration and exploitation, potentially leading to better solutions 
# and faster convergence compared to using a single metaheuristic alone.