# Name: spiral_hyperopt
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'constriction',
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses three different operators to find the solution: local_random_walk, spiral_dynamic and swarm_dynamic. 
# The local_random_walk operator is used with a probability of 0.75, a scale of 1.0 and a uniform distribution.
# The spiral_dynamic operator is used with a radius of 0.9, an angle of 22.5 and a sigma of 0.1.
# The swarm_dynamic operator is used with a factor of 0.7, a self confidence of 2.54, a swarm confidence of 2.56 and a constriction version.
# The metaheuristic uses the metropolis selection method to select the next solution.
# After running the metaheuristic for 30 times, we should get a smaller fitness solution.

fitness = []
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=50)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])