# Name: Hybrid Metaheuristic for Rastrigin Function (RastriginHybrid)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'random_sample',
        {},
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
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
# The metaheuristic named "RastriginHybrid" is a combination of different search operators designed to explore the Rastrigin function efficiently. 
# It starts with a random sampling operator to cover the entire space initially, followed by a spiral dynamic operator that navigates along a spiral path around the global optimum. This helps in refining the solution near the optimal region.
# Finally, a swarm dynamic operator is used to simulate a flocking behavior that helps in converging to the best solution more effectively. The use of 'all' selectors ensures that every agent can explore the search space thoroughly.
# The selection of operators and parameters has been chosen based on their effectiveness in handling high-dimensional optimization problems like Rastrigin. This hybrid approach leverages the strengths of each operator, making it robust and suitable for solving complex optimization tasks.