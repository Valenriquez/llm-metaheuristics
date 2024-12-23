# Name: Balanced Metaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(3)
prob = fun.get_formatted_problem()

heur = [
    ('random_search', {'scale': 0.1, 'distribution': 'uniform'}, 'greedy'),
    ('central_force_dynamic', {'gravity': 0.01, 'alpha': 0.02, 'beta': 1.5, 'dt': 0.5}, 'all'),
    ('differential_mutation', {'expression': 'rand-to-best-and-current', 'num_rands': 3, 'factor': 0.75}, 'metropolis'),
    ('firefly_dynamic', {'distribution': 'gaussian', 'alpha': 1.2, 'beta': 1.1, 'gamma': 90.0}, 'probabilistic')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
# met.run()

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Balanced Metaheuristic combines four operators with their respective selectors to balance exploration and exploitation. 
# 'random_search' helps in exploring the solution space, 'central_force_dynamic' promotes a more balanced approach, 
# 'differential_mutation' encourages combining exploration with exploitation, and 'firefly_dynamic' spreads solutions smoothly.
# The use of 'greedy', 'all', 'metropolis', and 'probabilistic' selectors dynamically adjusts the strategy based on the current state,
# ensuring a robust optimization process. Running multiple iterations helps in evaluating the effectiveness of the approach.