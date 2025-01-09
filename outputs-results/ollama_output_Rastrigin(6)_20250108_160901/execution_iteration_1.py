# Name: Hybrid Metaheuristic for Global Optimization

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)
prob = fun.get_formatted_problem()

# Define the search operators with their parameters and selectors
heur = [
    (  # Random Search Operator
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (  # Central Force Dynamic Operator
        'central_force_dynamic',
        {
            'gravity': 0.002,
            'alpha': 0.3,
            'beta': 1.8
        },
        'greedy'
    ),
    (  # Local Random Walk Operator
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 0.8,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (  # Swarm Dynamic Operator
        'swarm_dynamic',
        {
            'factor': 0.9,
            'self_conf': 2.6,
            'swarm_conf': 2.7,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
]

# Initialize the Metaheuristic with the problem and operators
met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
# met.verbose = True # Uncomment to enable verbose mode for debugging
# met.run() # Uncomment to run the metaheuristic

# Collect fitness data over multiple runs
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The hybrid metaheuristic combines the strengths of multiple search operators to improve global optimization capabilities. Each operator is chosen based on its suitability for different aspects of the problem.
# - Random Search Operator helps escape local minima.
# - Central Force Dynamic Operator encourages exploration with a balance between attraction and repulsion.
# - Local Random Walk Operator facilitates fine-tuning near optimal solutions.
# - Swarm Dynamic Operator simulates collective intelligence, enhancing search efficiency and robustness.
# The use of probabilistic selectors allows each operator to adaptively adjust its parameters based on historical performance. This hybrid approach aims to leverage the complementary advantages of different metaheuristic strategies, leading to more effective global optimization for complex problems like Rastrigin's function.