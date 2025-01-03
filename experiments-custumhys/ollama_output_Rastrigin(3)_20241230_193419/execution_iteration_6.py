# Name: Hybrid Metaheuristic Algorithm using Random Search and Firefly Dynamic
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
            'scale': 0.6922102169416083,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'firefly_dynamic',
        {
            'alpha': 0.1127616451886454,
            'beta': 1.001402210711944,
            'gamma': 83.90024318347375
        },
        'all'
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
# This hybrid metaheuristic combines the local search capabilities of Random Search with the global exploration power of Firefly Dynamic. The Random Search operator helps in fine-tuning the solution around promising regions, while the Firefly Dynamic operator facilitates a more extensive exploration of the solution space. The use of both operators ensures a balance between exploitation and exploration, leading to improved convergence towards optimal solutions.