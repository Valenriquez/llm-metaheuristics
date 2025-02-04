# Name: Hybrid Evolutionary Swarm Optimization with Modified Parameters (HESO)

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(3) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.6911845373419406,
            'self_conf': 2.2929937345697518,
            'swarm_conf': 2.126469820521639,
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
        'local_random_walk',
        {
            'probability': 0.5128970228971406,
            'scale': 0.9931891391938215,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
# met.verbose = True # please comment this line
# met.run() # please comment this line

fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Evolutionary Swarm Optimization with Modified Parameters (HESO) combines the strengths of swarm optimization, random sampling, and local search. It uses a modified version of the swarm dynamic operator to guide the agents towards potential solutions, while maintaining the exploration capabilities of random sample and fine-grained exploration with local random walk. These modifications aim to balance exploration and exploitation more effectively, potentially leading to better performance on benchmark problems like Cigar.