# Name: Diversified Metaheuristic

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
            'scale': 0.10607026127407493,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.0013546097497621485,
            'alpha': 0.014860930690299732,
            'beta': 1.2562543184770965,
            'dt': 1.48590529241125
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 1.9451361287315228
        },
        'metropolis'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 173
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
# This metaheuristic is designed to explore a diverse set of strategies during the search process. It starts with a simple `random_search` operator and gradually introduces more sophisticated operators like `central_force_dynamic`, `differential_mutation`, and `firefly_dynamic`. Each operator uses different parameters and selection methods, ensuring that the solution space is explored from various angles. This diversity helps in finding better solutions by leveraging the strengths of each operator while maintaining a balance between exploration and exploitation. The use of multiple operators also aids in handling different landscapes more effectively.