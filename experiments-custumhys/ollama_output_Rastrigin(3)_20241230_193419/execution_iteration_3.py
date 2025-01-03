# Name: HybridMetaheuristic

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_search',
        {
            'scale': 0.4792907508207271,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.007683876840395065,
            'alpha': 0.02699016024949217,
            'beta': 1.9983832168302882,
            'dt': 1.8987171674763907
        },
        'metropolis'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 142.99754121391692
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

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
# This hybrid metaheuristic combines three different search operators: Random Search, Central Force Dynamic, and Firefly Dynamic. Each operator is configured with its own set of parameters to explore the solution space effectively. The combination of these operators allows for a diverse exploration strategy, leveraging their unique characteristics to enhance the overall performance of the metaheuristic.