# Name: Spiral-Enhanced Swarm Dynamics (SASD)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (
        'spiral_dynamic',
        {
            'radius': 0.3391930382262049,
            'angle': 0.02173893846741559,
            'sigma': 0.055252790976172285
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.3170130190525082,
            'self_conf': 2.1000432571949075,
            'swarm_conf': 2.9778142360831876,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=1000)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# The chosen metaheuristic combines the Spiral Dynamic (SASD) operator with a Swarm Dynamics operator to optimize the Rastrigin function. 
# The Spiral Dynamic operator is designed to explore solutions in a spiral pattern, while the Swarm Dynamics operator facilitates exploration and exploitation through particle interactions.
# These combinations help balance between local search and global search, making the metaheuristic more effective in finding the optimal solution for the Rastrigin function.

# Feedback:
# It appears that there was an error in your code due to invalid syntax. The error message indicates a problem at line 1, suggesting that the hyperparameter tuning process you are using may not be effectively finding the best hyperparameters.
# I recommend reviewing your hyperparameters and adjusting them as needed to ensure the metaheuristic operates correctly.

# To fix the error, please check the code for any syntax issues or typos and make necessary corrections.