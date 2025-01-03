# Name: Hybrid Metaheuristic for Optimization Problems
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
    (  # Search operator 1: Random Search
        'random_search',
        {
            'scale': 0.03903712529724935,
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic', # Search operator 2: Central Force Dynamic
        {
            'gravity': 0.04725249709010224,
            'alpha': 0.5921935795341271,
            'beta': 1.2387415469845546,
            'dt': 1.9452164677033639
        },
        'probabilistic'
    ),
    (
        'firefly_dynamic', # Search operator 3: Firefly Dynamic
        {
            'distribution': 'gaussian',
            'beta': 1.2387415469845546
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
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=10)  
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The Hybrid Metaheuristic combines three different search operators to optimize the problem: Random Search, Central Force Dynamic, and Firefly Dynamic. This combination allows for a broad exploration of the solution space while also fine-tuning towards an optimal solution. The metropolis selector helps in escaping local minima by accepting worse solutions with a certain probability, improving the overall search efficiency. Running the metaheuristic 30 times ensures robustness and provides a good estimate of the best solution found across multiple initializations and runs.