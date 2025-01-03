# Name: Hybrid Metaheuristic with Diverse Operators and Selectors

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
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
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

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
# This hybrid metaheuristic combines three diverse operators: random_sample, swarm_dynamic, and local_random_walk.
# The 'random_sample' operator is used to diversify the search space by generating random samples.
# The 'swarm_dynamic' operator mimics the behavior of particle swarms, using a combination of cognitive (self-confidence) and social (swarm confidence) components.
# The 'local_random_walk' operator performs local searches using a probabilistic approach based on a Gaussian distribution.
# Each operator is paired with a selector to control the acceptance or rejection of solutions.
# The 'greedy' selector prioritizes immediate improvements, while the 'probabilistic' and 'metropolis' selectors allow for more exploratory behavior by accepting worse solutions with some probability.

# Addressed error:
# The error was due to an unexpected keyword argument 'factor' in the 'central_force_dynamic' operator. This parameter was removed from its definition in the heur list.