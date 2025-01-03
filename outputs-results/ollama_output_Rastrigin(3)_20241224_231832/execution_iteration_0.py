# Name: Hybrid Metaheuristic for Rastrigin Function

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
        'central_force_dynamic',
        {
            'gravity': 0.004375331860901789,
            'alpha': 0.2554658744567053,
            'beta': 1.849084747393388,
            'dt': 0.7389702407448977
        },
        'metropolis'
    ),
    (
        'genetic_crossover',
        {
            'pairing': 'cost',
            'crossover': 'two',
            'mating_pool_factor': 0.6656778911245328
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8434395607345041,
            'scale': 1.3944070167289289,
            'distribution': 'uniform'
        },
        'probabilistic'
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
    print('rep = {}, x_best = {}, f_best = {}'.format(rep + 1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The hybrid metaheuristic combines the Central Force Dynamic (CFD), Genetic Crossover (GC), and Local Random Walk (LRW) operators to explore the search space effectively.
# CFD helps in rapid convergence towards promising regions with a gravity of 0.004375, alpha of 0.2555, beta of 1.8491, and dt of 0.7390.
# GC ensures diversity by combining solutions using the 'cost' pairing method, two-point crossover, and a mating pool factor of 0.6657.
# LRW introduces randomness for escape from local minima with a probability of 0.8434, scale of 1.3944, and a uniform distribution.
# The metropolis selector is used for genetic crossover and probabilistic selector for local random walk to balance exploitation and exploration.