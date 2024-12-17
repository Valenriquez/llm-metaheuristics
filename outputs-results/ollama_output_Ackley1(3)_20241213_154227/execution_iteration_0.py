# Name: Spiral Dynamic Metaheuristic (SDM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin({self.dimensions})  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

num_agents = self.dimensions + 2  # Add more agents depending on the size of the dimension

heur = [
    (  # Search operator 1
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
            'factor': 0.7 if self.dimensions <= 10 else 0.8,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform' if self.dimensions <= 5 else 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=num_agents)
met.verbose = True
#met.run()

#print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The SDM combines Spiral Dynamic and Swarm Dynamic operators to explore the search space effectively. 
# The number of agents scales with the dimension to ensure adequate coverage.
# The use of probabilistic selectors helps balance exploration and exploitation throughout the iterations.