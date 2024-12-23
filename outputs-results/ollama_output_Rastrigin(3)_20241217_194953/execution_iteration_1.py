# Name: Hybrid Adaptive Metaheuristic (HAM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search with Adaptive Scaling
        'random_search',
        {
            'initial_scale': 0.3693425972851212,
            'scale_factor': 0.95,  # Decrease scale after each successful iteration
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        # Search operator 2: Central Force Dynamic with Adaptive Gravity
        'central_force_dynamic',
        {
            'gravity': 0.007509222662017407,
            'alpha': 0.002215583983671063,
            'beta': 1.0172747684482208,
            'dt': 0.2515441635118518,
            'gravity_decay': 0.98,  # Decrease gravity after each successful iteration
        },
        'metropolis'
    ),
    (
        # Search operator 3: Differential Mutation with Adaptive Factor
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 1.3242436287544117,
            'factor_decay': 0.9,  # Decrease mutation factor after each successful iteration
        },
        'uniform'
    ),
    (
        # Search operator 4: Local Search with Adaptive Perturbation
        'local_search',
        {
            'neighborhood_size': 5,
            'perturbation_factor': 0.1,
            'decay_rate': 0.99,  # Decrease perturbation factor after each successful iteration
        },
        'uniform'
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
# The Hybrid Adaptive Metaheuristic (HAM) combines multiple search operators with adaptive parameters to improve the search efficiency. 
# It starts with Random Search to explore the solution space widely and then uses Central Force Dynamic and Differential Mutation for more focused exploration.
# Local Search is added to refine solutions locally, with adaptive perturbation to escape local minima. 
# Each operator has its own set of parameters that adapt over time based on the success of the search operations, enhancing the overall performance of the algorithm.