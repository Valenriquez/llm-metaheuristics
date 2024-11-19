# Name: Hybrid Swarm and Spiral Dynamic Metaheuristic (HSSDM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # Changed this line to match the provided function and dimension.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all' # Changed selector to 'all' for better exploration in 3 dimensions.
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
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
# The Hybrid Swarm and Spiral Dynamic Metaheuristic (HSSDM) combines the advantages of swarm dynamic and spiral dynamic search operators. 
# Swarm dynamic is used to explore the solution space efficiently by simulating the behavior of particles in a group. Spiral dynamic helps refine the solutions by moving along spirals, which can be particularly useful in finding local optima.
# The 'all' selector for the swarm dynamic operator ensures that the search is thorough and covers a wide area of the solution space, especially important in higher dimensions like 3D. The probabilistic selector for spiral dynamic provides a balance between exploitation and exploration by sometimes choosing random moves.
# By combining these two operators, HSSDM aims to efficiently explore the Rastrigin function's complex landscape while effectively converging towards optimal solutions.