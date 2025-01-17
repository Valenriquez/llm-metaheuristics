# Name: Hybrid Metaheuristic Algorithm

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the operators and their parameters/selectors with updated values
heur = [
    (
        'random_search',
        {
            'scale': 0.06552392608519672,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0023709653571989528,
            'beta': 2.725989382503411
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7387253377840668,
            'self_conf': 2.509294019498595,
            'swarm_conf': 2.4116373375690507,
            'version': 'constriction'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8999901485220182,
            'angle': 21.28977222509448
        },
        'all'
    )
]

# Create an instance of the Metaheuristic class with the defined search operators and problem
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = False

# Run the metaheuristic with the same problem 30 times
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.run()
    
    fitness.append(met.historical['fitness'])

# Convert the fitness data to a NumPy array and extract the final fitness values
fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic Algorithm combines several search operators to explore the solution space more effectively. 
# 'random_search' is used to diversify the population, 'central_force_dynamic' simulates gravitational forces between particles,
# 'swarm_dynamic' mimics social behaviors with updated constriction parameters, and 'spiral_dynamic' employs a spiral trajectory for exploration.
# The combination of these operators allows for a robust search strategy that balances exploration and exploitation. 
# Running the algorithm multiple times helps in assessing its performance and convergence characteristics.