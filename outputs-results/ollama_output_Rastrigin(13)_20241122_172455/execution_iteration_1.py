# Name: Spiral and Swarm Dynamic Metaheuristic

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'spiral_dynamic',
        {
            'radius': 0.7478469702024668,
            'angle': 0.3933710400766669,
            'sigma': 0.4696807053187122
        },
        'probabilistic'
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
        'probabilistic'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines the Spiral Dynamic operator with the Swarm Dynamic operator to explore the search space effectively. The Spiral Dynamic helps in generating solutions that are spread out and diverse, while the Swarm Dynamic helps in converging quickly to the optimal solution. The combination ensures a balance between exploration and exploitation.

# With these values as parameters: {'radius': 0.7478469702024668, 'angle': 0.3933710400766669, 'sigma': 0.4696807053187122} and {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}.
# The error detected was due to an unterminated string literal at the beginning of the script. This has been fixed by ensuring all strings are properly enclosed in quotes.

# Feedback has been considered and implemented to resolve any syntax errors and improve the metaheuristic's functionality.