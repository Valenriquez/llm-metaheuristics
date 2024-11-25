# Name: Improved Swarm Dynamic Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)  # This is the selected problem, dimension provided.
prob = fun.get_formatted_problem()

heur = [
    (  
        'swarm_dynamic',
        {
            'factor': 0.28180536892289515,
            'self_conf': 0.13997387179411896,
            'swarm_conf': 0.852703493312263,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
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
# The Improved Swarm Dynamic Metaheuristic combines the swarm dynamic approach with specific parameter tuning to enhance its performance on high-dimensional optimization problems like the Rastrigin function. By adjusting the 'factor', 'self_conf', 'swarm_conf', and other parameters, we aim to achieve a balance between exploration and exploitation, leading to more efficient convergence to the global optimum.

# The greedy selector ensures that each iteration focuses on improving the best solution found so far, which can help in finding good solutions quickly while not getting stuck in local optima.

# By running the metaheuristic multiple times with different random initializations (controlled by 'num_agents'), we can obtain a more robust evaluation of its performance.