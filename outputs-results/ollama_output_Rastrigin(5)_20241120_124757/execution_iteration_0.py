# Name: CustomMetaheuristic

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # Example problem: Rastrigin function with dimension 5.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
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
        'greedy'
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
# This metaheuristic combines two search operators: `swarm_dynamic` and `spiral_dynamic`. 
# The `swarm_dynamic` operator is designed to simulate the behavior of a group of particles moving towards the best solution, while the `spiral_dynamic` operator uses a spiral pattern to explore the solution space.
# By combining these operators, we aim to leverage their strengths in exploration and exploitation phases of the optimization process.

# With these values as parameters: factor=0.7, self_conf=2.54, swarm_conf=2.56, version='inertial', distribution='uniform' for `swarm_dynamic`, radius=0.9, angle=22.5, sigma=0.1 for `spiral_dynamic`.
# These parameters were chosen to balance the exploration and exploitation capabilities of the operators in the context of the Rastrigin function with dimension 5.