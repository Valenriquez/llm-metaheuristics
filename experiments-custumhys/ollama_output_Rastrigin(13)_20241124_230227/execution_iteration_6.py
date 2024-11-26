# Name: Hybrid Swarm Dynamic Spiral Dynamic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    ('swarm_dynamic', 
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    ('spiral_dynamic', 
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
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
# The Hybrid Swarm Dynamic Spiral Dynamic combines the strengths of both swarm dynamic and spiral dynamic operators to optimize the Rastrigin function in 13 dimensions. The swarm dynamic operator helps in exploring the solution space effectively by considering the best known positions and velocities of agents, while the spiral dynamic operator guides the search towards promising regions using a spiral movement pattern. This combination is expected to lead to faster convergence and better performance on complex optimization problems like Rastrigin.

# In case there was an error, please fix it. The error mentioned seems to be related to an unterminated string literal in the feedback. Please ensure that all strings are properly closed with a matching quote and check for any missing characters or syntax errors.