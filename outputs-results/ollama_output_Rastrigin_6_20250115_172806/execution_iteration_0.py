# Name: Hybrid Metaheuristic for Rastrigin Function

# Code:
import sys
from pathlib import Path
import numpy as np
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6)
prob = fun.get_formatted_problem()

heur = [
    (  # random_search operator
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'differential_mutation',  # differential_mutation operator
        {
            'expression': 'rand-to-best-and-current',
            'num_ranks': 2,  # added key instead of parameter1
            'factor': 0.8  # example factor value for mutation
        },
        'metropolis'
    ),
    (
        'swarm_dynamic',  # swarm_dynamic operator
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
# met.verbose = True # please comment this line
# met.run() # please comment this line

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=98)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    #print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The selected metaheuristic combines random_search, differential_mutation, and swarm_dynamic.
# - `random_search` introduces exploration using a Gaussian distribution to perturb solutions.
# - `differential_mutation`, with 'rand-to-best-and-current', enhances exploitation by combining best-known solutions for effective search space traversal.
# - The use of 'metropolis' selector adds an acceptance criterion based on solution quality, aiding in escaping local optima.
# - `swarm_dynamic` leverages swarm intelligence principles to adjust agent positions dynamically, using parameters suited for balancing exploration and convergence.
# - All three operators are selected with a variety of strategies ('greedy', 'metropolis', 'all') for diverse search behaviors and solution evaluations.
# This combination aims to exploit the strengths of each approach while mitigating weaknesses, providing robust optimization performance on Rastrigin's multi-modal landscape.