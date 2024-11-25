# Name: SpiralMetaheuristic

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13) # Replace 'Rastrigin' with your chosen benchmark function and adjust dimension accordingly.
prob = fun.get_formatted_problem()

heur = [
    ('spiral_dynamic', {
        'radius': 0.6546310701552689,
        'angle': 1.9948927382555994,
        'sigma': 0.2184716553244452
    }, 'all'),
    ('spiral_dynamic', {
        'radius': 0.5,
        'angle': 22.5,
        'sigma': 0.1
    }, 'metropolis')
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
# The metaheuristic combines the Spiral Dynamic operator with different parameters to explore the search space more effectively. 
# The 'all' selector is used for exploitation, while the 'metropolis' selector allows for exploration by occasionally accepting worse solutions.
# Adjusting the radius, angle, and sigma can help balance between exploration and exploitation, potentially improving convergence.