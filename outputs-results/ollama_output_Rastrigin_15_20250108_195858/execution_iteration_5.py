# Name: HybridMetaheuristic
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
    (
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=2000)
# met.verbose = True
# met.run()

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=2000, num_agents=50)
    met.reset_historicals()
    met.verbose = False
    met.run()
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines multiple operators to leverage their strengths. 
# The `random_search` operator is used to explore the solution space randomly, while the `central_force_dynamic` operator simulates physical forces between agents to guide them towards better solutions. 
# The `swarm_dynamic` operator mimics the behavior of social groups, enhancing cooperation and exploration.
# The `local_random_walk` operator allows for fine-tuning around promising regions.
# This combination helps in efficiently exploring both the global and local optima, improving the overall performance on the Rastrigin function.