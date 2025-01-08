# Name: Improved Particle Swarm Optimization (IPSO)

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]  # Ensure the correct project directory is set
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.8309232619908188,
            'self_conf': 2.8401206880667185,
            'swarm_conf': 2.5030401628631234,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5223028497983582,
            'angle': 24.54147722028314,
            'sigma': 0.010243736188053076
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

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
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The Improved Particle Swarm Optimization (IPSO) combines the swarm dynamics with spiral motion to enhance exploration and exploitation phases. 
# This hybrid approach aims to improve convergence speed and solution quality on complex optimization problems like the Rastrigin function.
# The use of probabilistic selectors ensures that the search process remains stochastic, avoiding premature convergence.