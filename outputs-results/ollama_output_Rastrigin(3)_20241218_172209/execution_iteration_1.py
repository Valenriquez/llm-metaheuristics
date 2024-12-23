# Name: Dynamic Hybrid Metaheuristic

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
    (  # Search operator 1: Random Walk
        'random_sample',
        {
            'step_size': 0.2867320776672429,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',  # Search operator 2: Central Force Dynamic
        {
            'gravity': 0.08325110936417432,
            'alpha': 0.06804580018748035,
            'beta': 1.1061322102088278,
            'dt': 1.434971508644094
        },
        'metropolis'
    ),
    (
        'differential_mutation',  # Search operator 3: Differential Mutation
        {
            'expression': 'rand',
            'num_rands': 1,
            'factor': 1.045042331400605
        },
        'probabilistic'
    ),
    (
        'firefly_dynamic',  # Search operator 4: Firefly Dynamic
        {
            'distribution': 'uniform',
            'alpha': 0.7,
            'beta_min': 0.2,
            'gamma': 1.1061322102088278
        },
        'best'
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
# This metaheuristic combines four search operators: Random Walk, Central Force Dynamic, Differential Mutation, and Firefly Dynamic. 
# Each operator is applied iteratively to explore the solution space effectively.
# The 'random_walk' operator helps in exploring new areas of the search space randomly.
# The 'central_force_dynamic' operator simulates gravitational interactions between particles, guiding them towards the optimal solution.
# The 'differential_mutation' operator mimics the mutation process in evolutionary algorithms, introducing diversity and improving exploration.
# The 'firefly_dynamic' operator models the flashing behavior of fireflies to attract better solutions, enhancing exploitation capabilities.
# This combination allows for a robust search strategy that balances both exploration and exploitation, making it suitable for solving complex optimization problems.