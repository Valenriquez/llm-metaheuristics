# Name: Simple Random Search with Selective Restarts (SR-Select)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(5)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'greedy'),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=5) # Please add more agents depending on the size of the dimension.
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic uses a simple random search operator to explore the solution space. The 'random_sample' operator samples potential solutions uniformly at random.
# To improve the chances of finding a good solution, selective restarts are performed after each set of 100 iterations. Each restart resets the agent positions and allows them to search anew for the optimal solution.
# By running the metaheuristic multiple times (30 in this case), we can estimate the distribution of best fitness values obtained across different random initializations and iterations.
# The selection of 'greedy' as the selector means that only the first agent with the lowest fitness value is kept at each step, which simplifies implementation and speeds up convergence.