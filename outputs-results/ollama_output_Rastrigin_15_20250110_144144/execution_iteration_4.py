# Name: Hybrid Metaheuristic Algorithm (HMA)
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

# Define a hybrid metaheuristic using multiple operators and selectors with specified parameters
heur = [
    (
        'random_search',
        {
            'scale': 0.17149074202836695,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.10130338142980982,
            'beta': 2.6330362163173193
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7275019705703156,
            'self_conf': 2.53056648995854,
            'swarm_conf': 2.314786512539641,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 21.999715068009408,
            'sigma': 0.050039428292913204
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
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
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HMA combines several search operators (Random Search, Central Force Dynamics, Swarm Dynamics, and Spiral Dynamics) with different selection strategies (greedy, all, and probabilistic). This hybrid approach aims to leverage the strengths of each operator while mitigating their weaknesses. By running the metaheuristic multiple times and averaging the final fitness values, we can gain a more reliable estimate of the problem's solution quality.