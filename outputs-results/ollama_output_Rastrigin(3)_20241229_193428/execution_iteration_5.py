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
            'scale': 0.1,
            'distribution': 'uniform'
        },
        'metropolis'
    ),
    (
        'central_force_dynamic',
        {
            'gravity': 0.05,
            'alpha': 0.02,
            'beta': 1.5,
            'dt': 1.0
        },
        'all'
    ),
    (
        'firefly_dynamic',
        {
            'distribution': 'gaussian',
            'alpha': 0.5,
            'beta': 0.8,
            'gamma': 50.0
        },
        'probabilistic'
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
# This hybrid metaheuristic combines three different operators: Random Search, Central Force Dynamic, and Firefly Dynamic. Each operator is configured with specific parameters to enhance its performance on the Rastrigin function. The Metaheuristic class from the `metaheuristic` module is used to run these operators iteratively, with 'metropolis' and 'all' selectors for the first two operators, and 'probabilistic' selector for the third operator. Running the metaheuristic multiple times allows us to observe its robustness and effectiveness in solving the optimization problem.