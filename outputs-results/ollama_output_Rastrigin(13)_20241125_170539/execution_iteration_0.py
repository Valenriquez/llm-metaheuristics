# Name: Simple Evolutionary Algorithm (SEA)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13) # This is the selected problem.
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
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The metaheuristic "Simple Evolutionary Algorithm (SEA)" is a combination of two search operators: 'random_sample' and 'swarm_dynamic'. 
# The 'random_sample' operator randomly selects individuals from the population, providing diversity. 
# The 'swarm_dynamic' operator mimics the behavior of social insects like ants or bees, using parameters to control their movement (factor, self_conf, swarm_conf, version, distribution).
# The selection is done greedily and metropolis style respectively, enhancing the exploration and exploitation capabilities of the algorithm.
# The results are collected over 30 runs to evaluate the performance.