# Name: Hybrid Metaheuristic for Global Optimization
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
        'random_search',
        {
            'scale': 0.16782316578017248,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.008787216317672198,
            'alpha': 0.9955350525179258,
            'beta': 1.1431608758838219
        },
        'probabilistic'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.6619518457431194,
            'self_conf': 1.6940604308337424,
            'swarm_conf': 2.913611432697983,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.4851715793027957,
            'angle': 23.05646306362786,
            'sigma': 0.2042612270226123
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)
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
# This hybrid metaheuristic combines four different search operators to explore the solution space more effectively. The random search operator helps in exploring uncharted regions of the solution space, while the central force dynamic operator is used for exploitation around promising areas. The swarm dynamic operator mimics the behavior of social animals, enhancing cooperation and leadership among agents. Lastly, the spiral dynamic operator aids in fine-tuning the solutions by moving in a spiral pattern around known good points. This combination aims to balance exploration and exploitation, leading to more robust and efficient optimization results.