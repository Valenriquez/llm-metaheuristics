# Name: HybridMetaheuristic
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
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        # Search operator 2: Central Force Dynamic
        'central_force_dynamic',
        {
            'gravity': 0.001,
            'alpha': 0.01,
            'beta': 1.5,
            'dt': 1.0
        },
        'all'
    ),
    (
        # Search operator 3: Differential Mutation
        'differential_mutation',
        {
            'expression': 'rand-to-best-and-current',
            'num_rands': 2,
            'factor': 0.5
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
# The HybridMetaheuristic combines three different search operators to explore the solution space more effectively. 
# Random Search helps in escaping local optima by exploring solutions randomly.
# Central Force Dynamic simulates the gravitational attraction between particles, guiding the agents towards better solutions.
# Differential Mutation uses differences between individuals in the population to create new candidate solutions, enhancing diversity and exploration.
# The combination of these operators ensures a comprehensive search strategy that balances exploration and exploitation.