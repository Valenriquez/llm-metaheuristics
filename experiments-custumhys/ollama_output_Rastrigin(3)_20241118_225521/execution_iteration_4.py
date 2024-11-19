# Name: Random Flight with Local Search
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
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The proposed metaheuristic combines a Random Flight operator with a Local Random Walk operator. 
# Random Flight helps the algorithm explore the search space by taking large steps in random directions.
# After each step, a Local Random Walk is performed to exploit the neighborhood of the current solution.
# This hybrid approach allows the algorithm to balance exploration and exploitation effectively.
# The 'greedy' selector for Random Flight ensures that the first feasible solution is accepted immediately.
# The 'all' selector for Local Random Walk considers all potential moves in each iteration.