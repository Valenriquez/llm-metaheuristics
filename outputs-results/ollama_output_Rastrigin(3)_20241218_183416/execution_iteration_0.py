# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.002,
            'alpha': 0.02,
            'beta': 1.6,
            'dt': 1.5
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 2,
            'factor': 1.2
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8,
            'self_conf': 3.0,
            'swarm_conf': 2.7,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'greedy'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The HybridMetaheuristic combines four different search operators to explore the solution space more effectively. 
# The `random_search` operator provides fine-tuning by using a Gaussian distribution with a small scale.
# The `central_force_dynamic` operator helps in escaping local minima with higher influence factors.
# The `differential_mutation` operator enhances exploration by combining elements from the best and current solutions.
# The `swarm_dynamic` operator balances exploration and exploitation for complex landscapes.