# Name: Hybrid Metaheuristic
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
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.01,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.005,
            'alpha': 0.02,
            'beta': 1.5,
            'dt': 1.0
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
# This hybrid metaheuristic combines the strengths of Random Search and Central Force Dynamics. 
# The Random Search operator is used to explore the solution space randomly, helping to escape local minima.
# The Central Force Dynamics operator simulates gravitational forces between particles, guiding them towards better solutions.
# The Metropolis selector is used for both operators to ensure that worse solutions have a chance of being accepted,
# allowing for a more thorough exploration and exploitation of the search space.
# The combination of these operators aims to balance exploration and exploitation, potentially leading to better convergence and solution quality.