# Name: Hybrid Evolutionary Algorithm with Dynamic Operators

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
        'local_random_walk',
        {
            'probability': 0.726867381514521,
            'scale': 0.6096865764388474,
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.9413286159123742,
            'self_conf': 2.968470107568462,
            'swarm_conf': 2.9705133477993764,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8896036004674297,
            'angle': 23.88781636893203,
            'sigma': 0.35546268274581994
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)
#met.verbose = True # please comment this line
#met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Evolutionary Algorithm with Dynamic Operators combines several search operators to enhance exploration and exploitation capabilities. Each operator is selected based on its strengths: 
# - 'random_sample' provides initial solutions.
# - 'local_random_walk' allows for fine-grained adjustments around current solutions.
# - 'swarm_dynamic' mimics the behavior of particle swarms, balancing exploration and exploitation through historical information with a constriction factor version.
# - 'spiral_dynamic' encourages a more diverse search by moving in spirals with varying radii and angles. This dynamic approach helps avoid local optima and promotes global convergence.