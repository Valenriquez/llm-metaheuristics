# Name: Dynamic Multi-Objective Particle Swarm Optimization (DMOPSO)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.benchmark_function_6({self.dimensions}) # WRITE IT EXACTLY AS GIVEN, BE CAREFUL
prob = fun.get_formatted_problem()

heur = [
    (
        'gravitational_search',
        {
            'gravity': 0.2,
            'alpha': 0.15,
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 30,
            'sigma': 0.1,
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=500)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Dynamic Multi-Objective Particle Swarm Optimization (DMOPSO) is a modified version of the standard particle swarm optimization algorithm designed to handle multi-objective problems dynamically. It combines two search operators, gravitational_search and spiral_dynamic.
# Gravitational Search Algorithm provides global exploration capabilities by applying gravity-based forces among particles in the search space, which helps the swarm to converge faster towards optimal solutions from different regions.
# Spiral Dynamic is a local exploitation operator that changes its behavior based on distance decay parameters for better refinement of promising solutions. The combination of these two algorithms aims to balance exploration and exploitation effectively across multiple objectives.
# The selection strategies used here are 'metropolis' and 'probabilistic'. Metropolis method guides particles towards new positions based on an acceptance probability, while probabilistic methods decide the movement rules with certain probabilities for each iteration step.
# The number of iterations is set relatively high to allow the algorithm sufficient time to explore the search space effectively. Increasing verbosity helps in monitoring its performance and convergence characteristics during optimization runs.