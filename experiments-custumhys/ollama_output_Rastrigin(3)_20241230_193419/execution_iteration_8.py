# Name: Hybrid Metaheuristic for Global Optimization

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
    (
        'random_search',
        {
            'scale': 0.07618467237106213,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.3040063757587233,
            'alpha': 0.34900597661908556,
            'beta': 2.6308955180696776,
            'dt': 1.069386917678436
        },
        'greedy'
    ),
    (
        'differential_mutation',
        {
            'expression': 'current',
            'num_rands': 1,
            'factor': 2.468493330995613
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
# The Hybrid Metaheuristic combines three different operators: Random Search, Central Force Dynamic, and Differential Mutation. 
# This combination aims to leverage the strengths of each operator in exploring the search space more effectively.
# The use of 'metropolis' selector for Random Search helps in handling local minima by accepting worse solutions with a certain probability.
# The 'greedy' selector for Central Force Dynamic encourages quicker convergence towards potentially better solutions.
# The 'probabilistic' selector for Differential Mutation allows for exploration through random combinations of solutions.