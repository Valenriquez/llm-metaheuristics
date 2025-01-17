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

# Define the heuristic operators and selectors
heur = [
    (  # Search operator 1: Differential Mutation
        'differential_mutation',
        {
            'expression': 'current-to-best',
            'num_rands': 4,
            'factor': 0.21552604158327482
        },
        'probabilistic'
    ),
    (
        # Search operator 2: Gravitational Search Algorithm (GSA)
        'swarm_dynamic',
        {
            'self_conf': 2.1487148035312487,
            'swarm_conf': 2.025527297710027,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        # Search operator 3: Particle Swarm Optimization (PSO)
        'local_random_walk',
        {
            'probability': 0.568959143789191,
            'scale': 0.5715965070066269
        },
        'metropolis'
    )
]

# Initialize and run the metaheuristic
met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)
met.verbose = False
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Run the metaheuristic with the same problem 30 times and collect fitness values
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic Algorithm (HMA) combines three different search operators: Differential Mutation, Gravitational Search Algorithm (GSA), and Particle Swarm Optimization (PSO). Each operator is configured with specific parameters to enhance its performance in finding the global optimum of the Rastrigin function. The use of probabilistic, all-greedy, and Metropolis selectors allows for exploration and exploitation strategies during the search process. Running the metaheuristic multiple times helps in assessing the robustness and reliability of the solution obtained.