# Name: Enhanced Randomized Search with Spiral Dynamics and Swarm Intelligence
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
    ('random_sample', {}, 'greedy'),
    ('spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    ('swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'probabilistic'
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
# This metaheuristic combines three different search strategies to enhance the exploration and exploitation of the search space.
# The 'random_sample' operator provides a basic random search mechanism, which helps in exploring new areas of the solution space.
# The 'spiral_dynamic' operator uses a spiral movement strategy, guided by parameters like radius, angle, and sigma. This helps in fine-tuning the search process around promising regions.
# The 'swarm_dynamic' operator incorporates elements of swarm intelligence, using parameters like factor, self_conf, and swarm_conf to simulate collective behavior and enhance convergence.
# By combining these strategies, the algorithm can efficiently navigate through the complex Rastrigin function landscape, finding optimal solutions effectively.