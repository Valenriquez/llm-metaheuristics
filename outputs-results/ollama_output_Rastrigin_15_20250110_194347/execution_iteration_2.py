# Name: Randomized Search with Adaptive Operators and Selectors

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem.
prob = fun.get_formatted_problem()

# Define a set of search operators with their respective parameters and selectors
heur = [
    ('random_sample', {}, 'greedy'),
    ('local_random_walk', {'probability': 0.5848733836569255, 'scale': 0.7861743616069542, 'distribution': 'gaussian'}, 'probabilistic'),
    ('swarm_dynamic', {'factor': 0.5271449946770889, 'self_conf': 1.6525451615642774, 'swarm_conf': 2.5159225347298397, 'version': 'constriction', 'distribution': 'gaussian'}, 'adaptive'),
    ('spiral_dynamic', {'radius': 0.5528945608578105, 'angle': 21.73827179399039, 'sigma': 0.19618634775732788}, 'metropolis')
]

# Initialize the metaheuristic with the problem and operators
met = mh.Metaheuristic(prob, heur, num_iterations=1000)

# Initialize the fitness register to store results across multiple runs
fitness = []
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    fitness.append(met.historical['fitness'])

# Convert the fitness data into a numpy array for easier manipulation
fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])

print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This metaheuristic incorporates several search operators with varying parameters to explore the solution space. The use of 'random_sample' provides a broad exploration, while 'local_random_walk', 'swarm_dynamic', and 'spiral_dynamic' offer more focused searches that adapt based on the problem's characteristics. The 'greedy', 'probabilistic', and 'adaptive' selectors help in managing the search process by guiding or adapting the operator's behavior dynamically.
# The selection of operators and selectors is based on their suitability for different aspects of the optimization problem, leveraging the strengths of each component to enhance overall performance.
# Running the metaheuristic multiple times (30 iterations) helps in gathering a robust sample of results, providing insights into the reliability and effectiveness of the chosen approach.