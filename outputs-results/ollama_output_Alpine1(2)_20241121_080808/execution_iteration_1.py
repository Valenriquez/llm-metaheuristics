# Name: Hybrid Swarm-Inspired Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.alpine1(2) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.7914467490412056,
            'angle': 16.023558905438794,
            'sigma': 0.09642051121211145
        },
        'proportional'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.3980461229105199,
            'self_conf': 0.7122981189516797,
            'swarm_conf': 0.41354600220014226,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'proportional'
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
# This hybrid metaheuristic combines the Spiral Dynamic and Swarm Dynamic operators to leverage their strengths. The Spiral Dynamic operator helps in exploring the solution space efficiently by following a spiral path, while the Swarm Dynamic operator encourages exploration and exploitation through particle interactions.
# The parameters are chosen based on preliminary experiments and heuristic considerations.