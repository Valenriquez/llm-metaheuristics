# Name: Hybrid Metaheuristic for Rastrigin Function (Rastrigin(13))
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13)
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
     'probabilistic'),
    ('spiral_dynamic', 
     {
         'radius': 0.9,
         'angle': 22.5,
         'sigma': 0.1
     },
     'probabilistic')
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
# This hybrid metaheuristic combines the strengths of swarm dynamic and spiral dynamic search operators to explore and exploit the solution space efficiently for the Rastrigin function with 13 dimensions.
# The 'swarm_dynamic' operator helps in maintaining a diverse population and escaping local minima, while the 'spiral_dynamic' operator ensures that the particles move towards the global optimum by adapting their direction based on historical data.