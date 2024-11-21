# Name: Hybrid Metaheuristic with Spiral and Swarm Dynamics

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # This is the selected problem.
prob = fun.get_formatted_problem()

# Initialize the list of search operators
heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    )
]

# Create the Metaheuristic instance
met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialize the fitness register
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
# This hybrid metaheuristic combines the Spiral Dynamics Optimization (SDo) with the Particle Swarm Optimization (PSO).
# SDo is a nature-inspired optimization method that mimics the behavior of a spiral formation in nature.
# PSO, on the other hand, is a population-based stochastic optimization technique that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality.
# By combining these two methods, we leverage the global search capability of SDo and the local search capability of PSO to effectively explore and exploit the search space.
# The use of 'all' as the selector ensures that both operators are applied simultaneously, allowing for a more comprehensive search process.