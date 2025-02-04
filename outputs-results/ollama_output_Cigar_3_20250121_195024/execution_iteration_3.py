# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(1) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Sample
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5379951507174552,
            'self_conf': 2.2066179675032926,
            'swarm_conf': 2.908707377861898,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.0209118706642564,
            'alpha': 0.013626684817614582
        },
        'greedy'
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
# HybridMetaheuristic combines three search operators: Random Sample, Swarm Dynamic, and Gravitational Search. The use of multiple operators allows the algorithm to explore different regions of the solution space, potentially leading to better convergence and a higher likelihood of finding optimal solutions. The parameters for Swarm Dynamic and Gravitational Search have been updated to improve their performance on the Cigar function problem.