# Name: SpiralMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    ('spiral_dynamic', {
        'radius': 0.8,
        'angle': 25.0,
        'sigma': 0.1
    }, 'greedy'),
    ('random_sample', {}, 'greedy')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# SpiralMetaheuristic combines the SpiralDynamic operator to guide agents along a spiral path towards an optimal solution, and RandomSample for exploration. The use of a greedy selector ensures that the best solutions are retained at each step.
# This combination is particularly effective for high-dimensional problems like Rastrigin, which often require careful navigation and balance between exploitation and exploration.