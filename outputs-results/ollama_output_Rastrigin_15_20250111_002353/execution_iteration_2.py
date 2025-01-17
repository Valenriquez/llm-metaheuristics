# Name: Hybrid Metaheuristic for Optimization

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

heur = [
    ('random_sample', {}, 'greedy'),
    ('swarm_dynamic', {'factor': 0.612838170263184, 'self_conf': 2.4903857487364522, 'swarm_conf': 2.676860396673298, 'version': 'constriction', 'distribution': 'uniform'}, 'probabilistic'),
    ('local_random_walk', {'probability': 0.880410928007161, 'scale': 0.8642527224001965, 'distribution': 'uniform'}, 'greedy')
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)

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
# This hybrid metaheuristic combines the strengths of three different search operators: 
# `random_sample` for initial exploration, `swarm_dynamic` with specific parameters for guided convergence, and
# `local_random_walk` to escape local minima. The probabilistic selector ensures that each operator has a chance to contribute.
# Running the metaheuristic 30 times helps in assessing its robustness and reliability.