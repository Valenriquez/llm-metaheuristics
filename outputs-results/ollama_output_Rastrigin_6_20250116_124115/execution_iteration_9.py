# Name: Hybrid Metaheuristic for Rastrigin Function Optimization

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

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7440893174303383,
            'self_conf': 2.4292236731416685,
            'swarm_conf': 2.305606430366626,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'random_search',
        {
            'scale': 0.6720393559311139,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5638849196505099,
            'angle': 24.890362326117724,
            'sigma': 0.1742838431940262
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
# met.verbose = True
# met.run()

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic combines the strengths of different optimization techniques to enhance search efficiency and robustness. 
# 'swarm_dynamic' is used for global exploration, 'random_search' for local refinement, and 'spiral_dynamic' for fine-tuning.
# This combination helps in navigating through the complex landscape of the Rastrigin function effectively.