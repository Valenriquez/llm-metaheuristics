# Name: Randomized Search with Local Walks

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, Rastrigin function with 15 dimensions.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',  # Randomly sample initial population
        {},
        'greedy'
    ),
    (
        'local_random_walk',  # Perform local random walks
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True  # Uncomment to see verbose output
# met.run()  # Uncomment to run the metaheuristic

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This metaheuristic starts with a random sampling of the initial population. 
# It then uses local random walks to explore the search space, 
# allowing for a more thorough exploration around promising solutions. The Metropolis selector helps in accepting better or worse solutions 
# based on their fitness, thus avoiding getting stuck in local minima.