# Name: Hybrid Metaheuristic using Random Flight and Swarm Dynamics

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Random Flight
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'uniform',
            'beta': 1.5
        },
        'all'
    ),
    (
        'swarm_dynamic',  # Search operator 2: Swarm Dynamics
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
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
# The metaheuristic combines the random flight operator for exploring new regions of the search space with the swarm dynamic operator to exploit promising areas. Both operators are used together because they complement each other: random flight helps escape local minima, while swarm dynamics helps in converging to optimal solutions more efficiently.
# The 'all' selector is used for both operators as the dimensionality of the problem (5) is relatively low and there is no need for a more complex selection strategy.