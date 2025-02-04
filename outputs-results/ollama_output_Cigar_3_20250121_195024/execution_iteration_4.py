# Name: Hybrid Swarm-Local Search Metaheuristic

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
    (  # Search operator 1: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.5981238016558365,
            'self_conf': 2.7649950277225592,
            'swarm_conf': 2.623619560625613,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        # Search operator 2: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.9596856715752206,
            'scale': 1.5044291218085524,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)

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
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of a swarm dynamic operator and a local random walk.
# The swarm dynamic operator helps in exploring the global search space effectively, while the local random walk operator fine-tunes solutions around promising areas. 
# Together, this combination is expected to enhance the exploration-exploitation trade-off, leading to better convergence and performance on the Cigar function.