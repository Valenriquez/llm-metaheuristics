# Name: Hybrid Evolutionary Algorithm with Randomized Search

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
    ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'greedy'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 'all')
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
# This hybrid evolutionary algorithm uses two search operators: Spiral Dynamic (spiral_dynamic) and Swarm Dynamic (swarm_dynamic). 
# Spiral Dynamic is good for exploring the solution space by moving in a spiral pattern, which helps in finding global optima. 
# Swarm Dynamic simulates the behavior of particles in a flock or school, allowing for collective intelligence in searching the problem space.
# The 'greedy' selector is used with Spiral Dynamic to ensure that promising solutions are quickly adopted, while the 'all' selector with Swarm Dynamic allows all potential improvements to be considered collectively.
# By combining these operators, the metaheuristic aims to balance exploration and exploitation, leading to better convergence properties for solving Rastrigin's function in a 5-dimensional space.