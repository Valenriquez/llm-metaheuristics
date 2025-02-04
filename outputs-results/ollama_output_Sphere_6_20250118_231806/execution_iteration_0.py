# Name: Hybrid Dynamic Swarm Optimization (HDOS)

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(6)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.8818620414733611,
            'self_conf': 2.0110682287316384,
            'swarm_conf': 2.9968938253328465,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5019292340613488,
            'angle': 12.346597205663006,
            'sigma': 0.19714174602898227
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# HDOS combines the strengths of both swarm optimization and spiral dynamics. The swarm_dynamic operator helps in exploring the solution space effectively, while the spiral_dynamic operator aids in fine-tuning the solutions around local optima. This hybrid approach leverages the global search capabilities of swarm algorithms with the local refinement abilities of spiral dynamics, leading to a more robust and efficient metaheuristic for solving optimization problems like the Sphere function.