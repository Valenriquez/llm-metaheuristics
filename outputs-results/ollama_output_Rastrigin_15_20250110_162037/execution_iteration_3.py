# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.0505630042076315,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0026104583982490626,
            'alpha': 0.0321661574451113,
            'beta': 2.1269427602970716,
            'dt': 1.023851115432475
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6505238783531381,
            'self_conf': 1.0043255054261404,
            'swarm_conf': 1.4179294320521998,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8373711251266779,
            'angle': 21.17474953581598,
            'sigma': 0.19988327009330467
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The HybridMetaheuristic combines four different search operators to explore the solution space more efficiently. 
# 'random_search' provides initial exploration with a larger scale, 'central_force_dynamic' helps in moving towards better regions
# using an updated gravity and other parameters, 'swarm_dynamic' utilizes swarm intelligence for collective problem-solving with
# a constriction version and more agents, and 'spiral_dynamic' guides exploration in a structured manner.
# The use of different selectors like 'greedy', 'all', and 'metropolis' ensures that the algorithm can adaptively choose
# the best approach based on the current state of the solution. This combination aims to balance exploration and exploitation,
# leading to more robust and efficient optimization.