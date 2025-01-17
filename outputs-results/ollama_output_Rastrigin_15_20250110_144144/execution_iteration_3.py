# Name: Hybrid Metaheuristic

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
    ('random_search',
        {
            'scale': 0.03329486082554466,
            'distribution': 'uniform'
        },
        'greedy'),

    ('central_force_dynamic',
        {
            'gravity': 0.001604471458047079,
            'alpha': 0.09323527436000552,
            'beta': 3.4748133884506625,
            'dt': 1.480504965698449
        },
        'all'),

    ('swarm_dynamic',
        {
            'factor': 0.6549946043353777,
            'self_conf': 2.982286108793042,
            'swarm_conf': 2.5362264764868536,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'probabilistic')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# Hybrid Metaheuristic combines Random Search, Central Force Dynamic, and Swarm Dynamic operators.
# Each operator has been chosen based on its strengths in exploring the solution space effectively.
# The 'greedy' selector is used for Random Search to quickly find promising regions of the search space,
# while 'all' selector allows Central Force Dynamic to explore thoroughly across different dimensions.
# The probabilistic selector for Swarm Dynamic ensures a balance between exploration and exploitation.