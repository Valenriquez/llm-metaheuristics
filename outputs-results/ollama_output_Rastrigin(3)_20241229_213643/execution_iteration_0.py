# Name: HybridMetaheuristic
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
    (
        'random_search',
        {
            'scale': 0.5256457521611647,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.006514839907591067,
            'alpha': 0.011303780743743691,
            'beta': 2.945889273071346,
            'dt': 0.11487319850337913
        },
        'all'
    ),
    (
        'differential_mutation',
        {
            'expression': 'best',
            'num_rands': 1,
            'factor': 0.9824458432087539
        },
        'metropolis'
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of three different search operators: 
# 1. 'random_search' for exploring the solution space randomly,
# 2. 'central_force_dynamic' for guiding the search towards promising areas based on a force model,
# 3. 'differential_mutation' for making incremental improvements in the population.
# The combination is driven by greedy, all, and metropolis selectors to ensure diversity, exploration, and exploitation respectively. This approach aims to balance global and local search capabilities to effectively solve the Rastrigin function problem.