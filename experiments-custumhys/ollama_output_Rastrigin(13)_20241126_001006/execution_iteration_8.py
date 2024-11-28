# Name: Modified Inertial Swarm Optimization with Local Walks
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)  # Example problem with dimension 13.
prob = fun.get_formatted_problem()

# Increase number of agents based on dimensionality
num_agents = 15 * (13 - 2)

heur = [
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
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=2500, num_agents=num_agents) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines swarm optimization with local random walks to balance exploration and exploitation.
# The modified inertial version of the swarm updates positions using a velocity vector that is influenced by previous velocities.
# Local random walks are added to encourage exploration in areas where the solution space might be promising.
# By integrating these two techniques, we aim to achieve better convergence rates and escape local minima more effectively.