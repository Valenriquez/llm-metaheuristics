# Name: RastriginSearch

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
            'scale': 0.4092097712633707,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7057879493503175,
            'self_conf': 2.683090393749568,
            'swarm_conf': 2.995428781717998,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'roulette_wheel'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=205)
# met.verbose = True
# met.run()

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
# The RastriginSearch metaheuristic combines two search operators: `random_search` and `swarm_dynamic`.
# - The `random_search` operator is used to explore the solution space randomly, with a larger scaling factor to increase exploration.
# - The `swarm_dynamic` operator simulates the behavior of social animals using constriction version for better convergence while maintaining diversity.
# Both operators are selected with a metropolis selector for acceptance criteria and roulette wheel selection for choosing agents to apply the operators, respectively. The combination aims to balance exploration and exploitation effectively, leading to improved performance on the Rastrigin function optimization problem.