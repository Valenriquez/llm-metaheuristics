# Name: Hybrid Metaheuristic with Random Search and Differential Mutation
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
            'scale': 0.36614069935462035,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'differential_mutation',
        {
            'expression': 'current',
            'num_rands': 1,
            'factor': 0.513279170265837
        },
        'probabilistic'
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
# This hybrid metaheuristic combines Random Search (RS) and Differential Mutation (DM) to explore the solution space more effectively. 
# RS helps in exploring diverse regions of the search space due to its stochastic nature, while DM utilizes population-based operations for exploitation.
# The use of 'metropolis' selector allows the algorithm to accept worse solutions with a certain probability, aiding in avoiding local optima.
# The combination is expected to improve convergence rates and solution quality on the Rastrigin function.