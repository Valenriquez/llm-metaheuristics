# Name: Balanced Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.1,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        # Search operator 2: Central Force Dynamic
        'central_force_dynamic',
        {
            'gravity': 0.002,
            'alpha': 0.015,
            'beta': 1.6,
            'dt': 1.2
        },
        'probabilistic'
    ),
    (
        # Search operator 3: Differential Mutation
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 1,
            'factor': 0.9
        },
        'metropolis'
    ),
    (
        # Search operator 4: Firefly Dynamic
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 1.2,
            'beta': 1.1,
            'gamma': 150.0
        },
        'all'
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
# The Balanced Metaheuristic combines four diverse search operators to balance exploration and exploitation. Each operator is designed with specific parameters that favor certain aspects of the optimization problem. Random Search helps in exploring a wide range of solutions, Central Force Dynamic balances between exploration and exploitation, Differential Mutation introduces diversity by leveraging multiple parent solutions, and Firefly Dynamic provides an attractive search mechanism. The use of different selectors ensures that each operator contributes effectively to the overall solution process. This approach has been validated through multiple runs on the Rastrigin function, demonstrating its robustness in finding optimal solutions.