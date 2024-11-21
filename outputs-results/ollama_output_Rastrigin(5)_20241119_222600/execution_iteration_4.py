# Name: HybridMetaheuristic
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
    ('spiral_dynamic', 
     {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 
     'all'),
    ('swarm_dynamic', 
     {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 
     'all')
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
# The HybridMetaheuristic combines the Spiral Dynamic and Swarm Dynamic operators to explore the search space efficiently. 
# Spiral Dynamic helps in fine-grained exploration near the optimal solution, while Swarm Dynamic aids in broader exploration and exploitation.
# Using 'all' as the selector allows both operators to be used concurrently throughout the metaheuristic process, enhancing its robustness.
# The algorithm is run 30 times with varying iteration counts and agents to gather data on its performance and convergence rate.