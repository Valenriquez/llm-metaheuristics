# Name: Advanced Hybrid Metaheuristic (AHM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # Change the benchmark function and dimension here if needed.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
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
    ),
    (
        'random_sample',
        {},
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
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
# AHM is a hybrid metaheuristic that combines three different search operators to solve optimization problems.
# The 'spiral_dynamic' operator helps in exploring the solution space by spiraling towards the optima, which is useful for continuous optimization problems like Rastrigin.
# The 'swarm_dynamic' operator mimics the behavior of particles in a swarm to efficiently search and explore multiple regions simultaneously. It uses an inertial version, which keeps particles moving along their current path unless they encounter a better solution.
# The 'random_sample' operator provides random samples from the feasible space, ensuring thorough exploration and diversification. This helps in escaping local optima.
# The use of different selectors (greedy, all) for each operator allows them to operate effectively in diverse regions of the search space.
# By combining these operators, AHM aims to leverage their strengths and compensate for their weaknesses, leading to a more robust and efficient optimization process.