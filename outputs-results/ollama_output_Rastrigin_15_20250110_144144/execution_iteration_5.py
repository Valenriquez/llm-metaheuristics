# Name: Spiral Dynamics Optimization with Random Walks

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.6860121115175519,
            'angle': 24.94473867013677,
            'sigma': 0.2305565861847624
        },
        'metropolis'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.7778018930681543,
            'scale': 1.303093434723501,
            'distribution': 'levy'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=2000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The metaheuristic combines Spiral Dynamics Optimization with Random Walks to explore the search space more efficiently. The Spiral Dynamics operator helps in guiding the agents towards the optimal solution by simulating the dynamics of spiral motion, while the Local Random Walk operator ensures that the population does not get stuck in local optima.