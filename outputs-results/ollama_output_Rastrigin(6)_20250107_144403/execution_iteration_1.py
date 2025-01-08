# Name: Hybrid Metaheuristic for Rastrigin Function

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.6126665619321989,
            'angle': 22.76199960041798,
            'sigma': 0.010408075516450817
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6301917098196952,
            'self_conf': 2.0197885563468176,
            'swarm_conf': 2.0305807985072364,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
#met.verbose = True
#met.run()

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
# This hybrid metaheuristic combines three different search operators: random sample, spiral dynamic, and swarm dynamic. The random sample operator is used to initialize the population, while the spiral dynamic and swarm dynamic operators are used for exploration and exploitation respectively. The Metropolis selector is used for stochastic acceptance of new solutions in the spiral dynamic operator, and the probabilistic selector is used in the swarm dynamic operator to ensure a diverse set of solutions. This combination aims to balance exploration and exploitation while efficiently searching the solution space for the Rastrigin function.