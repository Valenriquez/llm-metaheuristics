# Name: Hybrid Metaheuristic with Random Search and Central Force Dynamic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the simplicity of random search with the efficient exploration capabilities of central force dynamic. 
# Random search is used to explore the solution space, while central force dynamics helps in converging towards the optimal solution more effectively.
# The 'greedy' selector ensures that the best solutions found during each iteration are retained, while the 'probabilistic' selector allows for a balance between exploration and exploitation.
# This combination aims to find good solutions efficiently by leveraging both global and local search strategies.