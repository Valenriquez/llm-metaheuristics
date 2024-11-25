# Name: Adaptive Spiral Swarm Hybrid Metaheuristic (ASSHM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.ChungReynolds(3)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.6621418284777919,
            'angle': 16.6183624705574,
            'sigma': 1.0166132650182251
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
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
# This hybrid metaheuristic combines the Spiral Dynamic and Swarm Dynamic operators to leverage their strengths in exploration and exploitation. The Spiral Dynamic operator helps in efficiently searching the space by dynamically adjusting its radius and angle, while the Swarm Dynamic operator aids in maintaining a balance among particles through self-confidence and swarm confidence parameters. This combination ensures that the algorithm can adaptively explore the search space effectively while balancing exploration and exploitation.

# For the given values of {'radius': 0.6621418284777919, 'angle': 16.6183624705574, 'sigma': 1.0166132650182251}, the adaptive parameters allow for fine-tuning the exploration and exploitation phases of the algorithm.

# Errors and Fixes:
# The provided code snippet had a syntax error at line 61 due to an incomplete statement or misplaced character. The error was corrected by ensuring that all statements are properly closed and correctly formatted.