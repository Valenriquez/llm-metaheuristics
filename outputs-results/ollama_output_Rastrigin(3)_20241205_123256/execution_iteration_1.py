# Name: Improved Spiral Dynamic Metaheuristic (ISDM)
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
    (  # Search operator 1
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        'random_sample',
        {},
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)  # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Improved Spiral Dynamic Metaheuristic (ISDM) combines the spiral dynamic operator with a random sampling operator to explore the solution space more effectively. 
# The spiral dynamic operator is used to guide the agents towards the global optimum by spiraling around it, while the random sampling operator helps in escaping local minima.
# The 'all' selector ensures that all search operators are applied equally during each iteration, promoting a diverse exploration of the solution space.
# This combination aims to improve the convergence speed and robustness of the metaheuristic for the Rastrigin function.