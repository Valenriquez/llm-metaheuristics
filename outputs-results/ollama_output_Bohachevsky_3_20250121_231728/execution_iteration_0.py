# Name: Bohachevsky Metaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(3) # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.010404089225021916,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.004501917090691247,
            'alpha': 0.05299526953210176,
            'beta': 1.500553315490849,
            'dt': 1.946447533150144
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
# This metaheuristic combines a random search operator with a central force dynamic operator. The random search helps in exploring the solution space randomly, while the central force dynamic operator guides the agents towards promising regions by simulating gravitational forces. The metropolis selector is used for both operators to allow exploration even if the new solutions are not immediately better. This combination aims to balance exploration and exploitation, leading to a more robust and effective search process for the Bohachevsky function.