# Name: Enhanced Hybrid Metaheuristic (EHM)
# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'Spiral Dynamics Optimization',
        {
            'step_size': 0.5,
            'learning_rate': 0.3
        },
        'Global'
    ),
    (
        'Local Random Walks',
        {
            'step_size': 0.1,
            'distribution': 'levy'
        },
        'Probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialize the fitness register
fitness = []
best_fitness = float('inf')

for rep in range(30):
    num_agents = 50
    met = mh.Metaheuristic(prob, heur, num_iterations=2000, num_agents=num_agents)
    met.reset_historicals()
    met.verbose = False
    met.run()

    fitness.append(met.historical['fitness'])
    
    # Adaptive population size based on convergence rate
    if best_fitness > met.f_best:
        best_fitness = met.f_best
        num_agents += 10  # Increase population size to explore further
    elif rep % 5 == 0:
        num_agents -= 10  # Reduce population size if not improving

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The EHM incorporates dynamic parameter tuning, adaptive population size, hybridization of global and local search strategies, and diversity control. By adjusting parameters and population size based on performance trends, the metaheuristic aims to enhance convergence and exploration capabilities.