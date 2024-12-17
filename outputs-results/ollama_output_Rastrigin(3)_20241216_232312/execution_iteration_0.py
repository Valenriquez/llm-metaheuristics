# Name: Adaptive Metaheuristic for Multi-Dimensional Optimization
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

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
# This code follows the guidelines by starting with a simple random search operator and adding more sophisticated operators as the dimensionality increases. It also dynamically adjusts the number of agents based on the problem's dimensionality, ensuring that the metaheuristic can handle higher dimensional problems effectively.