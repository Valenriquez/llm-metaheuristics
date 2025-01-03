# Name: Hybrid Metaheuristic for Rastrigin Function

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.694724555041569,
            'self_conf': 2.969029297978744,
            'swarm_conf': 2.961426505080846,
            'version': 'constriction',
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8050708253181967,
            'angle': 24.59150635168828,
            'sigma': 0.29472747496705537
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines three search operators: random_sample, swarm_dynamic, and spiral_dynamic. 
# The random_sample operator helps in exploring the solution space initially.
# The swarm_dynamic operator is used to guide the population towards better solutions using social behavior principles with the specified parameters.
# The spiral_dynamic operator adds a local exploration mechanism, helping to refine the solutions found by the global search operators with the specified parameters.
# By combining these operators, the hybrid metaheuristic aims to achieve a balance between exploration and exploitation, leading to better performance on the Rastrigin function.