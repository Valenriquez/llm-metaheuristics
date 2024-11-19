# Name: Customized Metaheuristic for Rastrigin Function with Dimension 5
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
    ('spiral_dynamic', {'radius': 0.9, 'angle': 18.0, 'sigma': 0.1}, 'all'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 'probabilistic')
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
# The metaheuristic combines the Spiral Dynamic and Swarm Dynamic operators to explore and exploit the search space for the Rastrigin function.
# - **Spiral Dynamic Operator**: It uses a spiral pattern to balance exploration and exploitation, starting with a high radius and gradually reducing it. This helps in converging to the global optimum efficiently.
# - **Swarm Dynamic Operator**: Utilizes particle swarm optimization principles to maintain a population of agents, each influenced by its own best position and the best position found by the group. The use of an inertial version ensures smoother convergence while maintaining diversity within the population.

# The combination of these operators allows for a robust search strategy that is effective in navigating the Rastrigin function's complex landscape.