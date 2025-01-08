# Name: Harmony Search with Adaptive Parameters

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

# Define operators with adaptive parameters
heur = [
    ('random_sample', {}, 'greedy'),
    ('spiral_dynamic', {'radius': 0.5000189893840966, 'angle': 15.489207292978199}, 'metropolis'),
    ('swarm_dynamic', {
        'factor': 0.30119994296486247,
        'self_conf': 2.9117413555896583,
        'swarm_conf': 2.8790186752720666,
        'version': 'inertial',
        'distribution': 'uniform'
    }, 'probabilistic'),
    ('random_sample', {}, 'greedy')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# Harmony Search (HS) is an optimization algorithm inspired by the improvisation process of musicians. The adaptive parameters allow the algorithm to explore different regions of the search space more effectively. The 'random_sample' operator introduces randomness, ensuring that new solutions are generated frequently. The 'spiral_dynamic' operator helps in escaping local minima by using a spiral motion strategy with specific radius and angle values. The 'swarm_dynamic' operator mimics the behavior of particles in a swarm with specified parameters for factor, self-confidence, swarm-confidence, version (inertial), and distribution type (uniform). The use of different selectors (greedy, metropolis, probabilistic) ensures that the algorithm can balance between exploration and exploitation, leading to improved convergence rates and better solutions.