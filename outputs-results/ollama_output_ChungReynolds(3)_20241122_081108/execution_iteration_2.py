# Name: CustomMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.ChungReynolds(3) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.36338603907261724,
            'angle': 19.752772788146448,
            'sigma': 0.8021628312549092
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.6,
            'swarm_conf': 3,
            'distribution': 'gaussian'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

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
# CustomMetaheuristic combines the spiral_dynamic operator with its specific parameters to guide the search in a spiraling pattern. The swarm_dynamic operator is used to enhance exploration by simulating a social behavior where agents adjust their velocities based on personal best and group consensus, using Gaussian distribution for smoother motion. This combination aims to balance between local exploitation and global exploration, potentially leading to more efficient optimization.