# Name: Multi-Agent Adaptive Search (MAAS)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'greedy'
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

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# MAAS combines a random sampling operator with a spiral dynamic operator to explore the search space efficiently. The use of the greedy selector for random sampling ensures that promising solutions are quickly adopted, while the Metropolis selector in the spiral dynamic operator allows for more thorough exploration by accepting worse solutions with a certain probability. This combination helps balance exploitation and exploration, leading to effective optimization.

# Note: For dimensions other than 3, you may need to adjust the number of agents and possibly tweak the parameters of the operators to ensure adequate performance.