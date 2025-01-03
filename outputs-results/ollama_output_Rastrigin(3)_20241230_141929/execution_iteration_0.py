# Name: Hybrid Metaheuristic using Random Search, Central Force Dynamic, and Differential Mutation
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.005,
            'alpha': 0.02,
            'beta': 1.4,
            'dt': 0.9
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 2,
            'factor': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=15)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of Random Search (local exploration), Central Force Dynamic (inertia-based global search),
# and Differential Mutation (population-based optimization). The selection of operators ensures a balance between exploration and exploitation.
# The random_search operator helps in finding good initial solutions, while central_force_dynamic and differential_mutation enhance
# convergence and solution quality. The use of different selectors further diversifies the search process, improving overall performance.