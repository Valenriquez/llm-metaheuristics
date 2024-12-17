# Name: Hybrid Search Metaheuristic (HSM)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.{self.benchmark_function}({self.dimensions}) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Genetic Algorithm
        'genetic',
        {
            'cxpb': trial(0.5, 'uniform'),
            'mutpb': trial(0.2, 'uniform')
        },
        'roulette'
    ),
    (
        'random_search', # Search operator 2: Random Search
        {
            'scale': trial(1.0, 'uniform'),
            'distribution': 'gaussian'
        },
        'uniform'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=self.dimensions) # Please add more agents depending on the size of the dimension. 
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Hybrid Search Metaheuristic (HSM) combines two search operators to leverage their respective strengths. 
# Genetic Algorithm: This operator uses evolutionary strategies like crossover (cxpb) and mutation (mutpb). 
# Random Search: This operator explores the solution space randomly, which can help escape local minima.
# By combining these operators, HSM aims to balance exploration and exploitation, potentially leading to better optimization results for complex benchmark functions.