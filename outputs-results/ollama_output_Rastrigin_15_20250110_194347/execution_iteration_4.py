# Name: Hybrid Metaheuristic Algorithm (HMA)
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

# Define the search operators and their parameters
heur = [
    (
        'random_search',
        {
            'scale': 0.03754584004594502,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.004447297341023438,
            'alpha': 0.062396505388299016,
            'beta': 1.8054988324880594
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.783001485136959,
            'self_conf': 2.7733652869830276,
            'swarm_conf': 2.685225593383218,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.7881773834542583,
            'angle': 23.689204807562895,
            'sigma': 0.08156556487853431
        },
        'greedy'
    )
]

# Initialize the metaheuristic with the problem and search operators
met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = False

# Run the metaheuristic for multiple repetitions
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.run()
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of various search operators tailored to different aspects of the Rastrigin function's characteristics. The 'random_search' operator helps escape local minima, while 'central_force_dynamic' and 'swarm_dynamic' provide global exploration capabilities. 'spiral_dynamic' ensures fine-tuning near the optimum. The use of different selectors allows for a balanced trade-off between exploitation and exploration throughout the search process.