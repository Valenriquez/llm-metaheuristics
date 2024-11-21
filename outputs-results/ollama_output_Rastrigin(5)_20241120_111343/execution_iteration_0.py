# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5) # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': "constriction",
            'distribution': "uniform"
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
# The hybrid metaheuristic combines the Spiral Dynamic and Swarm Dynamic operators to take advantage of their strengths.
# Spiral Dynamic is good at exploring the search space initially by spiraling outwards from a central point. 
# Swarm Dynamic, on the other hand, benefits from the collective behavior of a swarm, which can efficiently find and refine solutions.
# By using both operators in sequence (and potentially interleaving them), we aim to achieve a balance between exploration and exploitation,
# enhancing the overall performance and robustness of the metaheuristic.