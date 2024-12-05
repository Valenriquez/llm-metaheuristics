# Name: Randomized Search with Spiral Dynamics
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # Replace with the desired benchmark function and dimension.
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'greedy'),
    ('spiral_dynamic',
     {
         'radius': 0.9,
         'angle': 22.5,
         'sigma': 0.1
     },
     'probabilistic')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5)  # Adjust number of agents based on dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines a random sampling operator to explore the solution space widely and spiral dynamics to refine
# the search around promising regions. The use of probabilistic selection ensures that agents with better fitness have a higher
# chance of being selected for further exploration.