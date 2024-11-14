# Name: Rastriginn
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # Changed the problem to Rastrigin
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.8,
            'self_conf': 2.51,
            'swarm_conf': 2.55,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (  # Search operator 2
        'spiral_dynamic',
        {
            'radius': 0.91,
            'angle': 22.25,
            'sigma': 0.1
        },
        'metropolis'
    ),
    (  # Search operator 3
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three search operators: swarm dynamic, spiral dynamic, and local random walk. The swarm dynamic operator is used as the primary search strategy.
# It takes into account two parameters that control its behavior. The spiral dynamic operator provides a different type of search.
# The local random walk operator is used to avoid getting stuck in local optima.

fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=20) # Added num_agents=20 for better performance with higher dimension
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])