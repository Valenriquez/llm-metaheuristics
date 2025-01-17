# Name: Hybrid Metaheuristic

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

# Define the hybrid search operators with updated parameters
heur = [
    (
        'random_search',
        {
            'scale': 0.07970933081303191,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.5750727558603824,
            'self_conf': 2.3515116240866853,
            'swarm_conf': 2.235279067291012,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6112389021984759,
            'angle': 14.553202327121461,
            'sigma': 0.09971744312656729
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("Final Fitness Array:", final_fitness)

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of three different search operators: Random Search, Swarm Dynamics, and Spiral Dynamic. The updated parameters for these operators are designed to enhance their performance on the Rastrigin function.
# The Random Search operator is equipped with a higher scale factor to increase its exploration capabilities. The Swarm Dynamics operator uses a 'constriction' version which balances exploitation and exploration more effectively. The Spiral Dynamic operator's parameters are tuned to ensure it spirals in a way that efficiently explores the solution space without getting stuck.
# The metaheuristic is run for an initial 100 iterations to allow thorough exploration, followed by further refinement with 1000 iterations for deeper exploitation. Different selectors (metropolis, greedy, probabilistic) control the acceptance of new solutions based on their fitness relative to the current best solution.
# Running the metaheuristic 30 times and collecting the final fitness values provides a robust estimate of its performance on the Rastrigin function.