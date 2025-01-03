# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.1062109982109275,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.24991234608394003,
            'alpha': 0.7104964094089773,
            'beta': 3.3429779073104195,
            'dt': 2.5231967971775697
        },
        'metropolis'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 0.5127130738972296
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

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
# This hybrid metaheuristic combines three different search operators: 
# 1. Random Search for exploration,
# 2. Central Force Dynamic for guidance towards optima,
# 3. Differential Mutation for fine-tuning solutions.
# The 'greedy' selector is used for quick convergence, while 'metropolis' and 'probabilistic' selectors allow the algorithm to escape local minima and explore more of the solution space. This combination aims to balance exploration and exploitation effectively in solving the Rastrigin function problem.