# Name: Hybrid Metaheuristic for Rastrigin Function Optimization
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem.
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
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
    )
]

# Update the parameters for each operator
heur[1] = ('swarm_dynamic', {'factor': 0.3568344875136792, 'self_conf': 2.951976243005424, 'swarm_conf': 2.744741106507324, 'version': 'inertial', 'distribution': 'uniform'}, 'probabilistic')
heur[2] = ('spiral_dynamic', {'radius': 0.47507395521781026, 'angle': 5.677651051605848, 'sigma': 0.020831420146921104}, 'metropolis')

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

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
# This hybrid metaheuristic combines three different search operators: random sampling, swarm dynamics, and spiral dynamics. The `random_sample` operator is used to initialize the population randomly. The updated `swarm_dynamic` operator uses parameters tailored for better exploration of the Rastrigin function's landscape, including a uniform distribution instead of Gaussian. The `spiral_dynamic` operator has been given new parameters that might help in effectively escaping local optima. Each operator is configured with specific parameters to enhance the overall performance in optimizing the Rastrigin function.