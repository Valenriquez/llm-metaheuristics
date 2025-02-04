# Name: BohachevskyOptimizationMetaheuristic

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(3)  # This is the selected problem
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.34316621870032304,
            'self_conf': 2.3228496503732363,
            'swarm_conf': 2.8107266599298266,
            'version': 'constriction',
            'distribution': 'levy'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6290397809245795,
            'angle': 22.036932254436593,
            'sigma': 0.04655189667691042
        },
        'probabilistic'
    ),
    (
        'random_sample',
        {},
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=57)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The BohachevskyOptimizationMetaheuristic uses a combination of swarm_dynamic, spiral_dynamic, and random_sample operators to explore the search space. 
# Swarm_dynamic helps in collective exploration by updating agents based on their own best position and the best position found by any agent.
# Spiral_dynamic provides a more localized search around promising regions.
# Random_sample ensures that the algorithm does not get stuck in local minima by occasionally exploring random points in the search space.
# The metaheuristic runs for 1000 iterations with 57 agents, and the fitness is recorded over 30 independent runs to assess its performance.