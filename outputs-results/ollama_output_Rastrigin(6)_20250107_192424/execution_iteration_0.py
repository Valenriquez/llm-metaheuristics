# Name: Hybrid Metaheuristic Algorithm

# Code:

```python
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define the heuristics list
heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'uniform'
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
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic algorithm combines three different search operators: random search, swarm dynamic, and spiral dynamic. The random search operator helps to explore the solution space extensively, the swarm dynamic operator leverages the collective intelligence of a group for more informed exploration, and the spiral dynamic operator is designed to converge quickly towards optimal solutions. The metropolis selector is used for the random search operator to allow for uphill moves, while probabilistic and greedy selectors are used for the other operators to ensure diversity in the population and convergence.
```

# Short explanation and justification:
This hybrid metaheuristic algorithm combines three different search operators: random search, swarm dynamic, and spiral dynamic. The random search operator helps to explore the solution space extensively, the swarm dynamic operator leverages the collective intelligence of a group for more informed exploration, and the spiral dynamic operator is designed to converge quickly towards optimal solutions. The metropolis selector is used for the random search operator to allow for uphill moves, while probabilistic and greedy selectors are used for the other operators to ensure diversity in the population and convergence.