# Name: Hybrid Adaptive Metaheuristic (HAM)
# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5327574395900689,
            'self_conf': 2.5138475301025682,
            'swarm_conf': 2.761677262163555,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.5278982648005619,
            'angle': 22.678878206747896,
            'sigma': 0.12092398214596778
        },
        'probabilistic'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# This hybrid adaptive metaheuristic (HAM) combines three different search operators: random sample, swarm dynamic, and spiral dynamic. The random sample operator is used to initialize the population randomly, which helps in exploring the solution space widely. The swarm dynamic operator is incorporated with specific parameters ('factor': 0.5327574395900689, 'self_conf': 2.5138475301025682, 'swarm_conf': 2.761677262163555, 'version': 'constriction', 'distribution': 'levy') to simulate the behavior of particles in a swarm with enhanced convergence properties and a Levy flight distribution for exploration. The spiral dynamic operator adds another layer of exploration by simulating the movement of a spiral pattern with specific parameters ('radius': 0.5278982648005619, 'angle': 22.678878206747896, 'sigma': 0.12092398214596778), ensuring that all parts of the solution space are explored thoroughly while maintaining a balance between exploration and exploitation. This combination allows for a robust and adaptive search process, making HAM effective for solving complex optimization problems like the Rastrigin function with 15 dimensions.