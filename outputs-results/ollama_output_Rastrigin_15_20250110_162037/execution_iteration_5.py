# Name: Hybrid Metaheuristic for Rastrigin Function

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Walk
        'local_random_walk',
        {
            'probability': 0.8904348643293517,
            'scale': 1.1744797972192265,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',  # Search operator 2: Particle Swarm Optimization
        {
            'factor': 0.5712342490371567,
            'self_conf': 2.250360933034478,
            'swarm_conf': 2.8166234728286534,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)
# met.verbose = True  # please comment this line
# met.run()  # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines a local search operator (Random Walk) with a global search operator (Particle Swarm Optimization). The local walk helps in fine-tuning the solution, while PSO explores the solution space more extensively. The greedy selector ensures that the best solutions are retained at each step.