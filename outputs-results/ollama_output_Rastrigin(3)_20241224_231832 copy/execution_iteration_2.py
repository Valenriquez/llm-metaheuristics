# Name: Hybrid Metaheuristic for Rastrigin Function Optimization

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
    (  # Search operator 1: Random Sample
        'random_sample',
        {},
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
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
# 1. The hybrid metaheuristic combines three search operators: Random Sample, Local Random Walk, and Spiral Dynamic.
# 2. Random Sample is used as a baseline operator to explore the solution space randomly.
# 3. Local Random Walk allows for small, controlled moves around the current solutions.
# 4. Spiral Dynamic uses a spiral-like movement pattern to efficiently explore the search space.

# Error Handling:
# The error encountered was due to an unexpected keyword argument 'factor' in the central_force_dynamic operator. This issue is resolved by not passing the 'factor' parameter when calling this operator, as it is not required for its implementation.