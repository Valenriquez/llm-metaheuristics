# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search with Gaussian Distribution
        'random_search',
        {
            'num_agents': 205,
            'mean': prob.bounds.mean(),
            'std_dev': prob.bounds.std() / 3
        },
        'greedy'
    ),
    (
        # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        # Search operator 3: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
# met.verbose = True # please comment this line if needed for debugging
# met.run() # please comment this line if you want to run the metaheuristic

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# 1. HybridMetaheuristic combines different search operators to enhance exploration and exploitation.
# 2. The first operator uses random search with a Gaussian distribution centered around the problem's mean and scaled by the standard deviation divided by three.
# 3. The second operator employs swarm dynamic optimization, leveraging inertia and cognitive/confidence factors for better movement strategies.
# 4. The third operator incorporates spiral dynamics to help in finding local minima more effectively.
# 5. Running this metaheuristic multiple times ensures robustness and helps in capturing a wide range of potential solutions.