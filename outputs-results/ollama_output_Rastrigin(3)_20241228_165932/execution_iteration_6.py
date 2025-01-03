# Name: Harmony-Based Metaheuristic for Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 0.32056166821491816,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0018479048758909126,
            'alpha': 1.0848241616921106,
            'beta': 1.5018454692284402,
            'dt': 0.5178872559022469
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 1.212926376289332
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

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
# This metaheuristic combines three different search operators: 
# 1. 'random_search' to explore the solution space randomly,
# 2. 'central_force_dynamic' to simulate physical dynamics for exploration,
# 3. 'differential_mutation' to perform local searches around promising solutions.
# The combination of these operators helps in efficiently exploring and exploiting the search space, leading to better optimization results.