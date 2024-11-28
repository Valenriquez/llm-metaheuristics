# Name: Spiral Dynamic Metaheuristic (SDM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(13) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
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
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Spiral Dynamic Metaheuristic (SDM) combines the Spiral Dynamic operator with the Swarm Dynamic operator to explore the search space more effectively. The Spiral Dynamic operator helps in moving agents along a spiral path, which can help in escaping local minima. The Swarm Dynamic operator simulates the behavior of particles in a swarm, which can lead to better exploration and exploitation of the solution space. Both operators are used with a greedy selector to choose the best move at each step.

# By running the metaheuristic for 30 iterations with 2 agents, we ensure that the algorithm has enough time to converge to a good solution while maintaining diversity in the population. The use of both operators allows the algorithm to balance exploration and exploitation, making it suitable for solving high-dimensional optimization problems like Rastrigin's function.

# If there are any issues or errors, please address them as indicated above.