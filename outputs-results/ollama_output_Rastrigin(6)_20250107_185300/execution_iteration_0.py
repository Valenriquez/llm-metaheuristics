# Name: Hybrid Metaheuristic Algorithm

# Code:
import sys
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(6) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Define parameters for each operator
params_random_search = {
    'scale': 1.0,
    'distribution': 'uniform'
}

params_central_force_dynamic = {
    'gravity': 0.001,
    'alpha': 0.5,
    'beta': 1.5
}

params_local_random_walk = {
    'probability': 0.75,
    'scale': 1.0,
    'distribution': 'gaussian'
}

# Define the search operators and their selectors
heur = [
    ('random_search', params_random_search, 'greedy'),
    ('central_force_dynamic', params_central_force_dynamic, 'probabilistic'),
    ('local_random_walk', params_local_random_walk, 'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
# met.verbose = True # please comment this line
# met.run() # please comment this line

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=100, num_agents=100)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    # print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])

fitness_array = np.array(fitness).T
final_fitness = np.array([x[-1] for x in fitness_array.T])
print("final_fitness_array", final_fitness)

# Short explanation and justification:
# The Hybrid Metaheuristic Algorithm combines three different search operators: random search, central force dynamic, and local random walk. This combination aims to leverage the strengths of each operator. The random search is used to explore the solution space randomly, which helps in escaping local optima. The central force dynamic simulates a system with gravitational forces, guiding agents towards the best solutions. The local random walk allows for more fine-grained exploration around promising regions. By integrating these operators, the algorithm aims to find better solutions more efficiently.

# The use of different selectors (greedy, probabilistic, and all) ensures that the algorithm can adapt its behavior based on the problem's characteristics and the progress made during the search process. This flexibility helps in finding high-quality solutions even for complex benchmark problems like Rastrigin.