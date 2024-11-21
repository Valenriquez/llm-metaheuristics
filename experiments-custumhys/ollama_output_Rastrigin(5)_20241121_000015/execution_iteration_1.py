# Name: Hybrid Metaheuristic with Spiral Dynamic and Swarm Dynamic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.5405117232957318,
            'angle': 14.408693167938573,
            'sigma': 0.046498549162240936
        },
        'probabilistic'
    ),
    (  # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.11779253652571474,
            'self_conf': 2.951026835950703,
            'swarm_conf': 2.7883600653077947,
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines Spiral Dynamic and Swarm Dynamic operators. The Spiral Dynamic operator is used to explore the search space in a spiral manner, which can help escape local minima. The Swarm Dynamic operator is used to exploit the solutions found by other agents in the swarm, which can lead to faster convergence.
# The parameters for Spiral Dynamic are chosen based on previous studies that have shown good performance on Rastrigin's function. The parameters for Swarm Dynamic are also chosen based on previous studies that have shown good performance on Rastrigin's function.
