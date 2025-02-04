# Name: Adaptive Hybrid Metaheuristic for Cigar Benchmark Function

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Cigar(3) # This is the selected problem
prob = fun.get_formatted_problem()

# Define the hybrid search operators with the given parameters
heur = [
    (  # Search operator 1: Random Sample
        'random_sample',
        {},
        'greedy'
    ),
    (
        # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.6211520309178449,
            'self_conf': 2.5489438252547485,
            'swarm_conf': 2.299016823690115,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        # Search operator 3: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.2629160995336411,
            'scale': 1.955536419496148,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

# Initialize the metaheuristic with hybrid operators
met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)

# Run the metaheuristic and collect fitness results
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines three different search operators: Random Sample, Swarm Dynamic, and Local Random Walk. The parameters for the Swarm Dynamic operator have been adjusted as follows: {'factor': 0.6211520309178449, 'self_conf': 2.5489438252547485, 'swarm_conf': 2.299016823690115, 'version': 'constriction', 'distribution': 'uniform'}. The Local Random Walk operator has parameters: {'probability': 0.2629160995336411, 'scale': 1.955536419496148, 'distribution': 'gaussian'}. By combining these operators in a hybrid approach, we aim to balance exploration and exploitation effectively, potentially leading to better solutions for the Cigar benchmark function.