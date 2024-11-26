# Name: SpiralRandomSearch
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'uniform',
            'beta': 1.5
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The SpiralRandomSearch metaheuristic combines two search operators: Spiral Dynamic and Random Flight. Spiral Dynamic helps in navigating through the search space in a spiral manner, which can be useful for escaping local minima. Random Flight adds randomness to the search process, helping in exploring new areas of the solution space. The combination of both operators aims to balance exploitation (spiral navigation) with exploration (random flights), leading to potentially better optimization results.