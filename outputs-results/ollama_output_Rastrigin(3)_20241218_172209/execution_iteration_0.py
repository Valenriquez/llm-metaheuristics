# Name: Hybrid Metaheuristic

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
            'scale': 0.2867320776672429,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.08325110936417432,
            'alpha': 0.06804580018748035,
            'beta': 1.1061322102088278,
            'dt': 1.434971508644094
        },
        'metropolis'
    ),
    (
        'differential_mutation',
        {
            'expression': 'rand',
            'num_rands': 1,
            'factor': 1.045042331400605
        },
        'probabilistic'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'uniform',
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 198.91402407595777
        },
        'greedy'
    )
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
# The Hybrid Metaheuristic combines the strengths of different optimization techniques. It uses a combination of Random Search for exploration, Central Force Dynamic with Metropolis Selector for exploitation, Differential Mutation with Probabilistic Selector to enhance diversity, and Firefly Dynamic with Greedy Selector for local search refinement. This hybrid approach aims to balance between exploring new regions of the solution space and efficiently converging to optimal or near-optimal solutions.