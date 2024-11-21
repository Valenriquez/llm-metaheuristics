# Name: OptimizedPSO

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)
prob = fun.get_formatted_problem()

heur = [
    ('swarm_dynamic', {
        'factor': 0.8876630999253723,
        'self_conf': 3.0,
        'swarm_conf': 2.5,
        'version': 'constriction',
        'distribution': 'uniform'
    }, 'probabilistic'),
    ('spiral_dynamic', {
        'radius': 0.9,
        'angle': 22.5,
        'sigma': 0.1
    }, 'greedy')
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
# The chosen metaheuristic is a combination of the 'swarm_dynamic' and 'spiral_dynamic' operators. The 'swarm_dynamic' operator is well-suited for global search due to its swarm behavior, while the 'spiral_dynamic' operator helps in fine-tuning the solution around the best point found so far.
# These parameters were chosen based on previous runs and their ability to balance exploration and exploitation effectively.