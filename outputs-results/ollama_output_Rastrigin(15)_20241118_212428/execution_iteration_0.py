# Name: Hybrid Metaheuristic with Spiral Dynamic and Random Sample

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # Changed the function and dimension as specified.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'  # Using a larger selector for higher dimensionality.
    ),
    (
        'random_sample',
        {},
        'greedy'
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
# This hybrid metaheuristic combines the Spiral Dynamic operator with a Random Sample operator. 
# The Spiral Dynamic operator helps in efficiently searching through the space by spiraling outwards from an initial point,
# which is particularly useful for unimodal functions like Rastrigin's function.
# The Random Sample operator ensures that the search does not get stuck in local optima by exploring random points in the feasible region.
# Together, these operators enhance the exploration capabilities of the metaheuristic, leading to more effective global optimization.