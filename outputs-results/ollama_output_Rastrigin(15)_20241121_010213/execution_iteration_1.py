# Name: Hybrid Swarm Optimization Algorithm (HSOA)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'min_value': -5.12,
            'max_value': 5.12
        },
        'best_agent'
    ),
    (
        'swarm_dynamic',
        {
            'radius': 0.5692315463961984,
            'angle': 5.644685214616578,
            'sigma': 0.2159431653109221,
            'factor': 0.07664274316356223,
            'self_conf': 2.0094087459368195,
            'swarm_conf': 1.6269199950411628,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'best_agent'
    ),
    (
        'differential_evolution',
        {
            'F': 0.7,
            'CR': 0.8
        },
        'worst_agent'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=30) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid approach combines Random Search, Swarm Dynamic, and Differential Evolution to explore different aspects of the search space effectively. The parameters for each operator are chosen based on their proven performance in literature.